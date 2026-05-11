//
//  VpnController.swift
//  Runner
//
//  Bridge between Flutter (`gopher_eye/vpn_control` MethodChannel +
//  `gopher_eye/vpn_status` EventChannel) and the libzt-backed Packet Tunnel
//  Provider that joins the broker-managed ZeroTier network.
//
//  This file lives in the Runner target and is responsible for:
//    - exposing the device's ZeroTier node id (10 hex) to Dart
//    - creating/updating the NETunnelProviderManager (the iOS-visible profile)
//    - issuing connect/disconnect/getStatus over the MethodChannel
//    - streaming NEVPNStatusDidChange transitions back to Dart
//
//  The actual tunnel data plane lives in the Packet Tunnel Provider extension
//  target (`ZeroTierNetworkExtension`) — see ios/ZeroTierNetworkExtension/.

import Foundation
import Flutter
import NetworkExtension

/// Identifier that must match the Packet Tunnel Provider extension's bundle id
/// in the Xcode project. Keep in sync with
/// `ios/ZeroTierNetworkExtension/Info.plist`.
private let kTunnelProviderBundleId = "io.npcomplete.gophereye.PacketTunnel"

/// App Group identifier used to share files between the host app and the
/// tunnel extension (the libzt identity lives here so both processes see the
/// same node id).
private let kAppGroupId = "group.io.npcomplete.gophereye.vpn"

@objc final class VpnController: NSObject {
    private let methodChannel: FlutterMethodChannel
    private let eventChannel: FlutterEventChannel
    private var eventSink: FlutterEventSink?
    private var statusObserver: NSObjectProtocol?

    @objc init(messenger: FlutterBinaryMessenger) {
        self.methodChannel = FlutterMethodChannel(
            name: "gopher_eye/vpn_control",
            binaryMessenger: messenger
        )
        self.eventChannel = FlutterEventChannel(
            name: "gopher_eye/vpn_status",
            binaryMessenger: messenger
        )
        super.init()

        methodChannel.setMethodCallHandler { [weak self] call, result in
            self?.handle(call, result: result)
        }
        eventChannel.setStreamHandler(StreamHandler(controller: self))

        statusObserver = NotificationCenter.default.addObserver(
            forName: .NEVPNStatusDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.broadcastStatus()
        }
    }

    deinit {
        if let observer = statusObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    // MARK: - Dispatch

    private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "getStatus":
            loadManager { mgr in result(self.string(for: mgr?.connection.status)) }

        case "getNodeId":
            do {
                let nodeId = try ZeroTierIdentity.ensure(appGroupId: kAppGroupId)
                result(nodeId)
            } catch {
                result(FlutterError(code: "IDENTITY", message: "\(error)", details: nil))
            }

        case "installConfig":
            guard
                let args = call.arguments as? [String: Any],
                let networkId = args["networkId"] as? String
            else {
                result(FlutterError(
                    code: "BAD_ARGS",
                    message: "installConfig requires networkId",
                    details: nil
                ))
                return
            }
            let name = (args["tunnelName"] as? String) ?? "Gopher Eye"
            installConfig(networkId: networkId, displayName: name) { error in
                if let error = error {
                    if (error as NSError).code == NEVPNError.configurationReadWriteFailed.rawValue {
                        result(FlutterError(code: "PERMISSION_DENIED", message: error.localizedDescription, details: nil))
                    } else {
                        result(FlutterError(code: "INSTALL", message: error.localizedDescription, details: nil))
                    }
                } else {
                    result(nil)
                }
            }

        case "connect":
            connect { error in
                if let error = error {
                    if (error as NSError).code == NEVPNError.configurationReadWriteFailed.rawValue {
                        result(FlutterError(code: "PERMISSION_DENIED", message: error.localizedDescription, details: nil))
                    } else {
                        result(FlutterError(code: "CONNECT", message: error.localizedDescription, details: nil))
                    }
                } else {
                    result(nil)
                }
            }

        case "disconnect":
            disconnect { error in
                if let error = error {
                    result(FlutterError(code: "DISCONNECT", message: error.localizedDescription, details: nil))
                } else {
                    result(nil)
                }
            }

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - NETunnelProviderManager wiring

    private func loadManager(_ completion: @escaping (NETunnelProviderManager?) -> Void) {
        NETunnelProviderManager.loadAllFromPreferences { managers, _ in
            let mgr = (managers ?? []).first(where: { mgr in
                if let proto = mgr.protocolConfiguration as? NETunnelProviderProtocol {
                    return proto.providerBundleIdentifier == kTunnelProviderBundleId
                }
                return false
            })
            completion(mgr)
        }
    }

    private func installConfig(
        networkId: String,
        displayName: String,
        completion: @escaping (Error?) -> Void
    ) {
        loadManager { existing in
            let mgr = existing ?? NETunnelProviderManager()
            let proto = NETunnelProviderProtocol()
            proto.providerBundleIdentifier = kTunnelProviderBundleId
            // The extension reads this out of providerConfiguration on each
            // startTunnel; serverAddress is informational for the iOS UI.
            proto.serverAddress = displayName
            proto.providerConfiguration = [
                "networkId": networkId,
            ]

            mgr.protocolConfiguration = proto
            mgr.localizedDescription = displayName
            mgr.isEnabled = true

            mgr.saveToPreferences { error in
                // Apple's API requires loading again before connect can succeed.
                if let error = error {
                    completion(error); return
                }
                mgr.loadFromPreferences { loadError in
                    completion(loadError)
                }
            }
        }
    }

    private func connect(_ completion: @escaping (Error?) -> Void) {
        loadManager { mgr in
            guard let mgr = mgr else {
                completion(NSError(domain: "VpnController", code: -1,
                                   userInfo: [NSLocalizedDescriptionKey: "No tunnel profile installed"]))
                return
            }
            do {
                try mgr.connection.startVPNTunnel()
                completion(nil)
            } catch {
                completion(error)
            }
        }
    }

    private func disconnect(_ completion: @escaping (Error?) -> Void) {
        loadManager { mgr in
            mgr?.connection.stopVPNTunnel()
            completion(nil)
        }
    }

    // MARK: - Status streaming

    fileprivate func broadcastStatus() {
        loadManager { [weak self] mgr in
            guard let self = self, let sink = self.eventSink else { return }
            sink(self.string(for: mgr?.connection.status))
        }
    }

    fileprivate func setEventSink(_ sink: FlutterEventSink?) {
        self.eventSink = sink
        if sink != nil { broadcastStatus() }
    }

    private func string(for status: NEVPNStatus?) -> String {
        switch status {
        case .some(.invalid), .none: return "disconnected"
        case .some(.disconnected): return "disconnected"
        case .some(.connecting): return "connecting"
        case .some(.connected): return "connected"
        case .some(.reasserting): return "connecting"
        case .some(.disconnecting): return "disconnecting"
        @unknown default: return "disconnected"
        }
    }
}

private final class StreamHandler: NSObject, FlutterStreamHandler {
    weak var controller: VpnController?
    init(controller: VpnController) { self.controller = controller }

    func onListen(withArguments _: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        controller?.setEventSink(events)
        return nil
    }

    func onCancel(withArguments _: Any?) -> FlutterError? {
        controller?.setEventSink(nil)
        return nil
    }
}

// MARK: - ZeroTier identity

/// Resolves the device's ZeroTier node id (10 hex). The identity lives in the
/// shared App Group container so both the host app (this file) and the tunnel
/// extension see the same 10-hex node id.
///
/// libzt generates a fresh identity the first time the tunnel is started; we
/// surface that ID to Dart so it can be enrolled with the broker via the
/// server.
enum ZeroTierIdentity {
    enum IdentityError: Error, CustomStringConvertible {
        case missingContainer
        case notInitialized
        case readFailed(Error)

        var description: String {
            switch self {
            case .missingContainer:
                return "App Group container is unavailable; configure App Groups in Signing & Capabilities."
            case .notInitialized:
                return "ZeroTier identity has not been generated yet; bring the tunnel up once so libzt can mint one."
            case .readFailed(let err):
                return "Failed to read identity.public: \(err)"
            }
        }
    }

    /// Returns the 10-hex node id, generating one if necessary by asking the
    /// extension's libzt instance to initialize. In practice the file is
    /// already there after the first `connect()` call.
    static func ensure(appGroupId: String) throws -> String {
        guard let container = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroupId) else {
            throw IdentityError.missingContainer
        }
        let identityUrl = container.appendingPathComponent("zerotier/identity.public")
        do {
            let raw = try String(contentsOf: identityUrl, encoding: .utf8)
            // identity.public format is "<nodeid>:0:<publickey>"; we only need the node id.
            let nodeId = raw.split(separator: ":").first.map(String.init) ?? ""
            guard nodeId.count == 10 else { throw IdentityError.notInitialized }
            return nodeId.lowercased()
        } catch let error as IdentityError {
            throw error
        } catch {
            throw IdentityError.readFailed(error)
        }
    }
}
