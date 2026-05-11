//
//  PacketTunnelProvider.swift
//  ZeroTierNetworkExtension
//
//  Packet Tunnel Provider that brings up the ZeroTier data plane on-device.
//  The Runner target writes `networkId` into the `providerConfiguration` of
//  the NETunnelProviderProtocol; this extension reads it out, starts libzt
//  against the shared App Group storage, joins that network, and waits for
//  the controller to authorize the node and assign an IP before completing
//  startTunnel.
//
//  The extension runs in a separate process from the host app. It has access
//  to App Group storage (group.io.npcomplete.gophereye.vpn) and the shared
//  Keychain, but no UI affordances and no direct way to call back into Flutter.
//
//  Logs flow to the system log (Console.app, filtered by the extension's
//  bundle id) and to `os_log` at default level.
//
//  This file is the Swift entry point; the libzt integration itself
//  (initializing the node, joining the network, plumbing packets into the
//  NEPacketTunnelFlow) is configured in Xcode after adding the libzt SwiftPM
//  package. See README.md in this directory for one-time setup.

import Foundation
import NetworkExtension
import os

final class PacketTunnelProvider: NEPacketTunnelProvider {
    private static let log = OSLog(
        subsystem: "io.npcomplete.gophereye.PacketTunnel",
        category: "tunnel"
    )

    override func startTunnel(options: [String: NSObject]?,
                              completionHandler: @escaping (Error?) -> Void) {
        os_log("startTunnel", log: Self.log, type: .info)

        guard let proto = self.protocolConfiguration as? NETunnelProviderProtocol,
              let config = proto.providerConfiguration,
              let networkId = config["networkId"] as? String else {
            completionHandler(NSError(
                domain: "PacketTunnelProvider",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "missing networkId in providerConfiguration"]
            ))
            return
        }

        // Hand off to libzt: start the node, join the requested network, and
        // wait for the controller to assign this peer an IP. Once libzt
        // surfaces the assignment, configure the tunnel interface and
        // complete startTunnel. The ZeroTier controller is the source of
        // truth for IP assignment — we do not need an IP up front.
        ZeroTierRuntime.shared.start(networkId: networkId) { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .failure(let err):
                os_log("ZeroTier start failed: %{public}@", log: Self.log, type: .error, "\(err)")
                completionHandler(err)
            case .success(let assignment):
                os_log(
                    "ZeroTier joined %{public}@ as %{public}@",
                    log: Self.log,
                    type: .info,
                    networkId,
                    assignment.assignedIp
                )
                let settings = NEPacketTunnelNetworkSettings(tunnelRemoteAddress: assignment.assignedIp)
                let ipv4 = NEIPv4Settings(addresses: [assignment.assignedIp], subnetMasks: ["255.255.255.255"])
                ipv4.includedRoutes = assignment.routedDestinations.map {
                    NEIPv4Route(destinationAddress: $0, subnetMask: "255.255.255.255")
                }
                settings.ipv4Settings = ipv4
                settings.mtu = 2800  // ZeroTier's default L2 MTU

                self.setTunnelNetworkSettings(settings) { error in
                    if let error = error {
                        os_log("setTunnelNetworkSettings failed: %{public}@", log: Self.log, type: .error, "\(error)")
                    }
                    completionHandler(error)
                }
            }
        }
    }

    override func stopTunnel(with reason: NEProviderStopReason,
                             completionHandler: @escaping () -> Void) {
        os_log("stopTunnel reason=%{public}d", log: Self.log, type: .info, reason.rawValue)
        ZeroTierRuntime.shared.stop {
            completionHandler()
        }
    }
}

/// Thin facade around the libzt SDK. The actual integration is provided once
/// the libzt SwiftPM package is added to this extension target (see README).
/// Until then this is a no-op stub that keeps the file compilable so the host
/// app can be developed without blocking on the native dependency.
final class ZeroTierRuntime {
    /// IP assignment reported by libzt once the controller authorizes this
    /// node on the joined network.
    struct Assignment {
        /// IPv4 the controller assigned to this peer.
        let assignedIp: String
        /// Destinations to route through the tunnel. The data-plane peers in
        /// the same ZeroTier network advertise themselves once authorized;
        /// callers typically populate this with any addresses they want to
        /// reach via the tunnel.
        let routedDestinations: [String]
    }

    static let shared = ZeroTierRuntime()
    private init() {}

    func start(networkId: String, completion: @escaping (Result<Assignment, Error>) -> Void) {
        // TODO: replace with `zts_init_from_storage(...)`, `zts_node_start()`,
        // `zts_net_join(UInt64(networkId, radix: 16)!)`, then poll
        // `zts_net_get_status` / `zts_addr_get` until libzt reports an IP.
        // See: https://github.com/zerotier/libzt
        completion(.failure(NSError(
            domain: "ZeroTierRuntime",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey:
                "ZeroTierRuntime is a scaffold — wire libzt into this target per README.md"]
        )))
    }

    func stop(completion: @escaping () -> Void) {
        // TODO: `zts_node_stop()` then `zts_node_free()`.
        completion()
    }
}
