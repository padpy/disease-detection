import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/services.dart';
import 'package:gopher_eye/services/app_settings.dart';

/// Coarse-grained state surfaced by [VpnService.statusStream]. Mirrors the
/// NEVPNStatus values from iOS and adds a few in-app states that happen
/// before the OS-level tunnel ever exists (fetching the config from the
/// server, installing the profile, etc.).
enum VpnState {
  unknown,
  disabled,
  fetchingConfig,
  installingProfile,
  connecting,
  connected,
  disconnecting,
  permissionDenied,
  simulatorUnsupported,
  error,
}

class VpnException implements Exception {
  VpnException(this.message);
  final String message;
  @override
  String toString() => 'VpnException: $message';
}

/// Singleton facade over the native ZeroTier tunnel. The actual tunnel is
/// driven by `ios/Runner/VpnController.swift` over the
/// `gopher_eye/vpn_control` MethodChannel. Android is not implemented in this
/// pass — calls will throw [VpnException] on non-iOS platforms.
class VpnService {
  VpnService._();
  static final VpnService instance = VpnService._();

  static const _channel = MethodChannel('gopher_eye/vpn_control');
  static const _events = EventChannel('gopher_eye/vpn_status');

  final _stateController = StreamController<VpnState>.broadcast();
  StreamSubscription<dynamic>? _eventSub;
  VpnState _current = VpnState.unknown;
  String? _connectedAddress;

  VpnState get current => _current;
  String? get connectedAddress => _connectedAddress;
  Stream<VpnState> get statusStream => _stateController.stream;

  /// Should be called once at app boot. Subscribes to the native status
  /// stream, queries current state, and reflects it via [statusStream].
  Future<void> init() async {
    if (!_isSupported) {
      _set(VpnState.simulatorUnsupported);
      return;
    }
    _eventSub ??= _events.receiveBroadcastStream().listen(
          _onNativeEvent,
          onError: (_) => _set(VpnState.error),
        );
    try {
      final raw = await _channel.invokeMethod<String>('getStatus');
      _set(_parseState(raw));
    } on PlatformException catch (e) {
      _logDebug('init getStatus failed: $e');
      _set(VpnState.error);
    }
  }

  /// Turn the tunnel on. Reads the configured ZeroTier network id from
  /// [AppSettings] (no server contact is required — ZeroTier handles peer
  /// discovery itself), installs the iOS VPN profile (system permission
  /// prompt appears the first time), and connects.
  ///
  /// Persists `vpn_enabled=true` only on success so a crash mid-enable does
  /// not cause an auto-reconnect storm at next launch.
  Future<void> enable() async {
    if (!_isSupported) {
      _set(VpnState.simulatorUnsupported);
      throw VpnException('VPN is only supported on iOS device builds.');
    }

    final networkId = await AppSettings.getZtNetworkId();
    if (!RegExp(r'^[0-9a-f]{16}$').hasMatch(networkId)) {
      _set(VpnState.error);
      throw VpnException(
        'ZeroTier network id must be 16 hex characters (current: "$networkId").',
      );
    }

    try {
      _set(VpnState.installingProfile);
      await _channel.invokeMethod('installConfig', {
        'networkId': networkId,
        'tunnelName': 'Gopher Eye',
      });

      _set(VpnState.connecting);
      await _channel.invokeMethod('connect');

      await AppSettings.setVpnEnabled(true);
    } on PlatformException catch (e) {
      if (e.code == 'PERMISSION_DENIED') {
        _set(VpnState.permissionDenied);
      } else {
        _logDebug('enable failed: $e');
        _set(VpnState.error);
      }
      rethrow;
    }
  }

  /// Turn the tunnel off and persist `vpn_enabled=false`.
  Future<void> disable() async {
    if (!_isSupported) return;
    await AppSettings.setVpnEnabled(false);
    try {
      _set(VpnState.disconnecting);
      await _channel.invokeMethod('disconnect');
    } on PlatformException catch (e) {
      _logDebug('disable failed: $e');
      _set(VpnState.error);
    }
  }

  /// Block until the tunnel is fully up, or [timeout] elapses. Returns true
  /// if connected, false on timeout. Useful as a gate before the first API
  /// call on cold start when `vpn_enabled` was previously true.
  Future<bool> waitUntilReady({Duration timeout = const Duration(seconds: 5)}) async {
    if (_current == VpnState.connected) return true;
    if (!_isSupported) return false;
    try {
      await statusStream
          .firstWhere((s) => s == VpnState.connected)
          .timeout(timeout);
      return true;
    } on TimeoutException {
      return false;
    }
  }

  void _onNativeEvent(dynamic event) {
    if (event is String) {
      _set(_parseState(event));
    } else if (event is Map) {
      final status = event['status'] as String?;
      _connectedAddress = event['address'] as String? ?? _connectedAddress;
      if (status != null) _set(_parseState(status));
    }
  }

  void _set(VpnState s) {
    if (_current == s) return;
    _current = s;
    _stateController.add(s);
  }

  static VpnState _parseState(String? raw) {
    switch (raw) {
      case 'disconnected':
        return VpnState.disabled;
      case 'connecting':
        return VpnState.connecting;
      case 'connected':
        return VpnState.connected;
      case 'disconnecting':
        return VpnState.disconnecting;
      case 'permission_denied':
        return VpnState.permissionDenied;
      case 'simulator_unsupported':
        return VpnState.simulatorUnsupported;
      default:
        return VpnState.unknown;
    }
  }

  bool get _isSupported => !kIsWebStub && Platform.isIOS;

  void _logDebug(String message) {
    if (kDebugMode) {
      // ignore: avoid_print
      print('[VpnService] $message');
    }
  }
}

// Standalone constant so `Platform` import does not trip on web targets that
// don't have dart:io. Flutter strips dart:io on web — gating with this avoids
// an analyzer warning if the team ever builds for web (the app doesn't today).
const bool kIsWebStub = bool.fromEnvironment('dart.library.html');
