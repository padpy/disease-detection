/// Shape of the ZeroTier config returned by the server's `/vpn/config`
/// endpoint (which proxies the broker's admin API). Mirrors the
/// `PeerConfig` pydantic model in `broker/app/models.py`.
class ZeroTierConfig {
  ZeroTierConfig({
    required this.peerId,
    required this.nodeId,
    required this.networkId,
    required this.assignedIp,
    required this.serverIp,
  });

  /// e.g. ``user:abc123`` (server's peer identifier) or ``server``.
  final String peerId;

  /// This device's ZeroTier node id (10 hex chars).
  final String nodeId;

  /// The ZeroTier network the device should join (16 hex chars).
  final String networkId;

  /// Bare IPv4 the broker reserved for this device, e.g. ``10.66.66.7``.
  final String assignedIp;

  /// The application server's address on the ZeroTier network.
  final String serverIp;

  factory ZeroTierConfig.fromJson(Map<String, dynamic> json) {
    return ZeroTierConfig(
      peerId: json['peer_id'] as String,
      nodeId: json['node_id'] as String,
      networkId: json['network_id'] as String,
      assignedIp: json['assigned_ip'] as String,
      serverIp: json['server_ip'] as String,
    );
  }
}
