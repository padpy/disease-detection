import 'dart:async';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:gopher_eye/screens/qr_creator_screen.dart';
import 'package:gopher_eye/screens/validation_screen.dart';
import 'package:gopher_eye/services/api_client.dart';
import 'package:gopher_eye/services/app_settings.dart';
import 'package:gopher_eye/services/auth_service.dart';
import 'package:gopher_eye/services/sync_service.dart';
import 'package:gopher_eye/services/vpn_service.dart';

/// Single-pane settings UI. Covers everything that's user-configurable today:
/// server URL + sync, where detection runs, and which LLM the chat uses
/// (with separate model fields per provider). All values flow through
/// [AppSettings], which is the only place that talks to SharedPreferences.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _serverUrlController = TextEditingController();
  final _userController = TextEditingController();
  final _openAiKeyController = TextEditingController();
  final _openAiModelController = TextEditingController();
  final _serverModelController = TextEditingController();
  final _ztNetworkIdController = TextEditingController();

  bool _loading = true;
  bool _saving = false;
  bool _obscureKey = true;

  DetectionLocation _detectionLocation = DetectionLocation.local;
  bool _syncEnabled = false;
  LlmProvider _llmProvider = LlmProvider.openai;

  // VPN
  bool _vpnEnabled = false;
  VpnState _vpnState = VpnService.instance.current;
  StreamSubscription<VpnState>? _vpnSub;
  bool _vpnBusy = false;

  // Connectivity probe + LLM model discovery.
  bool? _serverReachable;
  bool _probing = false;
  List<String> _availableServerModels = const [];
  bool _loadingModels = false;
  String? _modelsError;

  @override
  void initState() {
    super.initState();
    _hydrate();
    _vpnSub = VpnService.instance.statusStream.listen((state) {
      if (!mounted) return;
      setState(() => _vpnState = state);
    });
  }

  @override
  void dispose() {
    _serverUrlController.dispose();
    _userController.dispose();
    _openAiKeyController.dispose();
    _openAiModelController.dispose();
    _serverModelController.dispose();
    _ztNetworkIdController.dispose();
    _vpnSub?.cancel();
    super.dispose();
  }

  Future<void> _hydrate() async {
    // Fan out independent reads in parallel — each is a platform-channel
    // round-trip (and `getOpenAiApiKey` hits the iOS keychain), so doing
    // them sequentially adds hundreds of ms before the form renders.
    final results = await Future.wait([
      AppSettings.getServerUrl(),
      AppSettings.getUserName(),
      AppSettings.getDetectionLocation(),
      AppSettings.getSyncEnabled(),
      AppSettings.getLlmProvider(),
      AppSettings.getOpenAiApiKey(),
      AppSettings.getOpenAiModel(),
      AppSettings.getServerLlmModel(),
      AppSettings.getVpnEnabled(),
      AppSettings.getZtNetworkId(),
    ]);
    final url = results[0] as String?;
    final user = results[1] as String;
    final detection = results[2] as DetectionLocation;
    final sync = results[3] as bool;
    final provider = results[4] as LlmProvider;
    final openAiKey = results[5] as String?;
    final openAiModel = results[6] as String;
    final serverModel = results[7] as String;
    final vpn = results[8] as bool;
    final ztNetworkId = results[9] as String;
    if (!mounted) return;
    setState(() {
      _serverUrlController.text = AppSettings.isServerUrlFromBuild
          ? (url ?? '')
          : (url ?? '');
      _userController.text = user;
      _detectionLocation = detection;
      _syncEnabled = sync;
      _llmProvider = provider;
      _openAiKeyController.text =
          AppSettings.isOpenAiApiKeyFromBuild ? '' : (openAiKey ?? '');
      _openAiModelController.text = openAiModel;
      _serverModelController.text = serverModel;
      _vpnEnabled = vpn;
      _ztNetworkIdController.text = ztNetworkId;
      _loading = false;
    });
    if ((url ?? '').isNotEmpty) {
      unawaited(_refreshModels());
    }
  }

  Future<void> _save() async {
    setState(() => _saving = true);
    try {
      if (!AppSettings.isServerUrlFromBuild) {
        await AppSettings.setServerUrl(_serverUrlController.text);
      }
      await AppSettings.setUserName(_userController.text);
      await AppSettings.setDetectionLocation(_detectionLocation);
      await AppSettings.setSyncEnabled(_syncEnabled);
      await AppSettings.setLlmProvider(_llmProvider);
      if (!AppSettings.isOpenAiApiKeyFromBuild) {
        await AppSettings.setOpenAiApiKey(_openAiKeyController.text.trim());
      }
      await AppSettings.setOpenAiModel(_openAiModelController.text.trim());
      await AppSettings.setServerLlmModel(_serverModelController.text.trim());
      try {
        await AppSettings.setZtNetworkId(_ztNetworkIdController.text);
      } on FormatException catch (e) {
        if (!mounted) return;
        ScaffoldMessenger.of(context)
          ..hideCurrentSnackBar()
          ..showSnackBar(SnackBar(content: Text('ZeroTier network id: ${e.message}')));
        return;
      }
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(const SnackBar(content: Text('Settings saved')));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _testConnection() async {
    if (_probing) return;
    // Make sure the URL the user just typed is the one we probe.
    if (!AppSettings.isServerUrlFromBuild) {
      await AppSettings.setServerUrl(_serverUrlController.text);
    }
    setState(() {
      _probing = true;
      _serverReachable = null;
    });
    final ok = await ApiClient.instance.ping();
    if (!mounted) return;
    setState(() {
      _probing = false;
      _serverReachable = ok;
    });
    if (ok) unawaited(_refreshModels());
  }

  Future<void> _refreshModels() async {
    if (_loadingModels) return;
    setState(() {
      _loadingModels = true;
      _modelsError = null;
    });
    try {
      if (!AppSettings.isServerUrlFromBuild) {
        await AppSettings.setServerUrl(_serverUrlController.text);
      }
      final models = await ApiClient.instance.listLlmModels();
      if (!mounted) return;
      setState(() {
        _availableServerModels = models;
        _loadingModels = false;
        // Auto-select if the picker has a default that isn't in the list.
        if (_serverModelController.text.isEmpty && models.isNotEmpty) {
          _serverModelController.text = models.first;
        }
      });
    } on ApiNotConfiguredException {
      if (!mounted) return;
      setState(() {
        _availableServerModels = const [];
        _loadingModels = false;
        _modelsError = 'Set a server URL to discover models.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _availableServerModels = const [];
        _loadingModels = false;
        _modelsError = 'Failed to list models: $e';
      });
    }
  }

  /// True when the device is not currently on the ZeroTier network. Used to
  /// gate the "Join network" button so it can only fire from a clean
  /// disconnected state.
  bool get _ztJoinable {
    if (_vpnBusy) return false;
    switch (_vpnState) {
      case VpnState.disabled:
      case VpnState.error:
      case VpnState.permissionDenied:
      case VpnState.unknown:
        return true;
      case VpnState.connecting:
      case VpnState.connected:
      case VpnState.disconnecting:
      case VpnState.fetchingConfig:
      case VpnState.installingProfile:
      case VpnState.simulatorUnsupported:
        return false;
    }
  }

  /// Save whatever the user has typed into the network-id field, then ask
  /// [VpnService] to bring the tunnel up. The toggle stays in sync with the
  /// result via [_vpnSub].
  Future<void> _joinZtNetwork() async {
    if (!_ztJoinable) return;
    final raw = _ztNetworkIdController.text.trim();
    try {
      await AppSettings.setZtNetworkId(raw);
    } on FormatException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text('ZeroTier network id: ${e.message}')));
      return;
    }
    setState(() {
      _vpnBusy = true;
      _vpnEnabled = true;
    });
    try {
      await VpnService.instance.enable();
    } catch (e) {
      final actual = await AppSettings.getVpnEnabled();
      if (!mounted) return;
      setState(() => _vpnEnabled = actual);
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text('Join failed: $e')));
    } finally {
      if (mounted) setState(() => _vpnBusy = false);
    }
  }

  Future<void> _toggleVpn(bool enable) async {
    if (_vpnBusy) return;
    setState(() {
      _vpnBusy = true;
      _vpnEnabled = enable;
    });
    try {
      if (enable) {
        await VpnService.instance.enable();
      } else {
        await VpnService.instance.disable();
      }
    } catch (e) {
      // Revert the toggle so it reflects what actually happened.
      final actual = await AppSettings.getVpnEnabled();
      if (!mounted) return;
      setState(() => _vpnEnabled = actual);
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text('VPN: $e')));
    } finally {
      if (mounted) setState(() => _vpnBusy = false);
    }
  }

  String _vpnStatusLine() {
    switch (_vpnState) {
      case VpnState.connected:
        final addr = VpnService.instance.connectedAddress;
        return addr == null ? 'Connected' : 'Connected as $addr';
      case VpnState.connecting:
        return 'Connecting…';
      case VpnState.disconnecting:
        return 'Disconnecting…';
      case VpnState.fetchingConfig:
        return 'Fetching config from server…';
      case VpnState.installingProfile:
        return 'Installing VPN profile (allow the prompt)…';
      case VpnState.permissionDenied:
        return 'Permission denied. Re-enable to retry.';
      case VpnState.simulatorUnsupported:
        return 'VPN is unavailable on the iOS simulator.';
      case VpnState.error:
        return 'Error. Tap the toggle to retry.';
      case VpnState.disabled:
        return 'Disconnected';
      case VpnState.unknown:
        return 'Status unknown';
    }
  }

  Color _vpnStatusColor() {
    switch (_vpnState) {
      case VpnState.connected:
        return Colors.greenAccent;
      case VpnState.permissionDenied:
      case VpnState.error:
        return Colors.redAccent;
      case VpnState.simulatorUnsupported:
        return Colors.amberAccent;
      default:
        return Colors.white70;
    }
  }

  Future<void> _pullSamples() async {
    try {
      final inserted = await SyncService.instance.pullSamples();
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text('Pulled $inserted new samples')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text('Pull failed: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: const Text('Settings', style: TextStyle(color: Colors.white)),
        actions: [
          TextButton(
            onPressed: _loading || _saving ? null : _save,
            child: Text(
              _saving ? 'Saving…' : 'Save',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: Colors.white))
          : SafeArea(
              child: ListView(
                padding: const EdgeInsets.symmetric(vertical: 16),
                children: [
                  _SectionHeader('Account'),
                  _Card(children: const [_AccountTile()]),
                  _SectionHeader('Server'),
                  _Card(children: [
                    if (AppSettings.isServerUrlFromBuild)
                      const _ReadOnlyHint(
                        'Server URL is set at build time and cannot be edited.',
                      )
                    else
                      _TextRow(
                        label: 'Base URL',
                        controller: _serverUrlController,
                        hint: 'http://192.168.1.42:5555',
                        keyboardType: TextInputType.url,
                      ),
                    _TextRow(
                      label: 'User name (optional)',
                      controller: _userController,
                      hint: 'Used as sample.user when syncing',
                    ),
                    Row(
                      children: [
                        FilledButton.tonal(
                          onPressed: _probing ? null : _testConnection,
                          child: _probing
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : const Text('Test connection'),
                        ),
                        const SizedBox(width: 12),
                        if (_serverReachable == true)
                          const _StatusChip(
                            color: Colors.greenAccent,
                            label: 'Reachable',
                            icon: Icons.check_circle_outline,
                          )
                        else if (_serverReachable == false)
                          const _StatusChip(
                            color: Colors.redAccent,
                            label: 'Not reachable',
                            icon: Icons.error_outline,
                          ),
                      ],
                    ),
                  ]),
                  _SectionHeader('VPN (ZeroTier)'),
                  _Card(children: [
                    SwitchListTile(
                      title: const Text(
                        'Tunnel server traffic through ZeroTier',
                        style: TextStyle(color: Colors.white),
                      ),
                      subtitle: const Text(
                        'Fetches a per-user config from the server, joins the '
                        'broker-managed ZeroTier network, and routes backend '
                        'traffic to the server peer.',
                        style: TextStyle(color: Colors.white60, fontSize: 12),
                      ),
                      value: _vpnEnabled,
                      onChanged: _vpnBusy ? null : _toggleVpn,
                      activeThumbColor: Colors.white,
                      activeTrackColor: Colors.white24,
                    ),
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                      child: Row(
                        children: [
                          Icon(
                            _vpnState == VpnState.connected
                                ? Icons.lock_outline
                                : Icons.lock_open,
                            size: 16,
                            color: _vpnStatusColor(),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _vpnStatusLine(),
                              style: TextStyle(
                                color: _vpnStatusColor(),
                                fontSize: 12,
                              ),
                            ),
                          ),
                          if (_vpnBusy)
                            const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white70,
                              ),
                            ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          TextField(
                            controller: _ztNetworkIdController,
                            style: const TextStyle(
                              color: Colors.white,
                              fontFamily: 'Menlo',
                            ),
                            decoration: const InputDecoration(
                              labelText: 'ZeroTier network ID',
                              labelStyle: TextStyle(color: Colors.white70),
                              hintText: '16 hex characters',
                              hintStyle: TextStyle(color: Colors.white38),
                              helperText:
                                  'ZeroTier needs only a network ID — no server '
                                  'URL required. Tap Join to bring the tunnel up.',
                              helperStyle:
                                  TextStyle(color: Colors.white54, fontSize: 11),
                              helperMaxLines: 2,
                              border: OutlineInputBorder(),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Colors.white24),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Colors.white),
                              ),
                            ),
                            keyboardType: TextInputType.text,
                            autocorrect: false,
                            enableSuggestions: false,
                            maxLength: 16,
                          ),
                          const SizedBox(height: 4),
                          OutlinedButton.icon(
                            onPressed: _ztJoinable ? _joinZtNetwork : null,
                            icon: const Icon(Icons.cable, size: 18),
                            label: Text(
                              _vpnState == VpnState.connected
                                  ? 'Joined'
                                  : 'Join network',
                            ),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: Colors.white,
                              disabledForegroundColor: Colors.white38,
                              side: BorderSide(
                                color: _ztJoinable
                                    ? Colors.white
                                    : Colors.white24,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ]),
                  _SectionHeader('Backup & Sync'),
                  _Card(children: [
                    SwitchListTile(
                      title: const Text(
                        'Sync samples to server',
                        style: TextStyle(color: Colors.white),
                      ),
                      subtitle: const Text(
                        'Mirrors every capture (and detection results) to '
                        'the configured server.',
                        style: TextStyle(color: Colors.white60, fontSize: 12),
                      ),
                      value: _syncEnabled,
                      onChanged: (v) => setState(() => _syncEnabled = v),
                      activeThumbColor: Colors.white,
                      activeTrackColor: Colors.white24,
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 6),
                      child: Row(
                        children: [
                          OutlinedButton.icon(
                            onPressed: _pullSamples,
                            icon: const Icon(Icons.cloud_download_outlined,
                                size: 18),
                            label: const Text('Pull from server'),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: Colors.white,
                              side: const BorderSide(color: Colors.white24),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ]),
                  _SectionHeader('Sample tags'),
                  _Card(children: [
                    ListTile(
                      leading: const Icon(
                        Icons.qr_code_2,
                        color: Colors.white,
                      ),
                      title: const Text(
                        'Create sample QR code',
                        style: TextStyle(color: Colors.white),
                      ),
                      subtitle: const Text(
                        'Encode location, name, id, and notes into a QR code '
                        'you can print and re-scan from the camera screen.',
                        style: TextStyle(color: Colors.white60, fontSize: 12),
                      ),
                      trailing: const Icon(
                        Icons.chevron_right,
                        color: Colors.white54,
                      ),
                      onTap: () => Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const QrCreatorScreen(),
                        ),
                      ),
                    ),
                  ]),
                  _SectionHeader('Detection'),
                  _Card(children: [
                    for (final loc in DetectionLocation.values)
                      RadioListTile<DetectionLocation>(
                        value: loc,
                        groupValue: _detectionLocation,
                        title: Text(
                          loc.label,
                          style: const TextStyle(color: Colors.white),
                        ),
                        subtitle: Text(
                          loc == DetectionLocation.local
                              ? 'Run YOLO + SAM + classifier on-device. Default.'
                              : 'Upload to the server and wait for it to '
                                  'return labels and masks.',
                          style: const TextStyle(
                              color: Colors.white60, fontSize: 12),
                        ),
                        onChanged: (v) {
                          if (v != null) {
                            setState(() => _detectionLocation = v);
                          }
                        },
                        activeColor: Colors.white,
                      ),
                  ]),
                  _SectionHeader('Chatbot'),
                  _Card(children: [
                    for (final p in LlmProvider.values)
                      RadioListTile<LlmProvider>(
                        value: p,
                        groupValue: _llmProvider,
                        title: Text(
                          p.label,
                          style: const TextStyle(color: Colors.white),
                        ),
                        subtitle: Text(
                          p == LlmProvider.openai
                              ? 'OpenAI API (text-only). Requires an API key.'
                              : 'Server-hosted multimodal model at '
                                  '/v1/chat/completions.',
                          style: const TextStyle(
                              color: Colors.white60, fontSize: 12),
                        ),
                        onChanged: (v) {
                          if (v != null) setState(() => _llmProvider = v);
                        },
                        activeColor: Colors.white,
                      ),
                    if (_llmProvider == LlmProvider.openai) ...[
                      const _Divider(),
                      if (AppSettings.isOpenAiApiKeyFromBuild)
                        const _ReadOnlyHint(
                          'OpenAI key is provided at build time and cannot be '
                          'edited from the app.',
                        )
                      else
                        _TextRow(
                          label: 'OpenAI API key',
                          controller: _openAiKeyController,
                          hint: 'sk-...',
                          obscure: _obscureKey,
                          onToggleObscure: () =>
                              setState(() => _obscureKey = !_obscureKey),
                        ),
                      _TextRow(
                        label: 'OpenAI model',
                        controller: _openAiModelController,
                        hint: AppSettings.defaultOpenAiModel,
                      ),
                    ] else ...[
                      const _Divider(),
                      _ServerModelPicker(
                        controller: _serverModelController,
                        models: _availableServerModels,
                        loading: _loadingModels,
                        error: _modelsError,
                        onRefresh: _refreshModels,
                      ),
                    ],
                  ]),
                  _SectionHeader('Validation'),
                  _Card(children: [
                    ListTile(
                      leading: const Icon(
                        Icons.tune,
                        color: Colors.white,
                      ),
                      title: const Text(
                        'Calibrate FHB HSV thresholds',
                        style: TextStyle(color: Colors.white),
                      ),
                      subtitle: const Text(
                        'Load an image from the photo library, run the '
                        'wheat-head segmentation pipeline, and tune the '
                        'green / necrotic HSV bands manually.',
                        style: TextStyle(color: Colors.white60, fontSize: 12),
                      ),
                      trailing: const Icon(
                        Icons.chevron_right,
                        color: Colors.white54,
                      ),
                      onTap: () => Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const ValidationScreen(),
                        ),
                      ),
                    ),
                  ]),
                  const SizedBox(height: 24),
                ],
              ),
            ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader(this.text);
  final String text;
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Text(
        text.toUpperCase(),
        style: const TextStyle(
          color: Colors.white60,
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.8,
        ),
      ),
    );
  }
}

class _Card extends StatelessWidget {
  const _Card({required this.children});
  final List<Widget> children;
  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (int i = 0; i < children.length; i++) ...[
            Padding(
              padding: const EdgeInsets.all(8),
              child: children[i],
            ),
          ]
        ],
      ),
    );
  }
}

class _Divider extends StatelessWidget {
  const _Divider();
  @override
  Widget build(BuildContext context) =>
      const Divider(color: Colors.white12, height: 1);
}

class _TextRow extends StatelessWidget {
  const _TextRow({
    required this.label,
    required this.controller,
    this.hint,
    this.keyboardType,
    this.obscure = false,
    this.onToggleObscure,
  });

  final String label;
  final TextEditingController controller;
  final String? hint;
  final TextInputType? keyboardType;
  final bool obscure;
  final VoidCallback? onToggleObscure;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 8, 8, 8),
      child: TextField(
        controller: controller,
        keyboardType: keyboardType,
        obscureText: obscure,
        autocorrect: false,
        style: const TextStyle(color: Colors.white),
        decoration: InputDecoration(
          labelText: label,
          labelStyle: const TextStyle(color: Colors.white60),
          hintText: hint,
          hintStyle: const TextStyle(color: Colors.white30),
          enabledBorder: const OutlineInputBorder(
            borderSide: BorderSide(color: Colors.white24),
          ),
          focusedBorder: const OutlineInputBorder(
            borderSide: BorderSide(color: Colors.white),
          ),
          suffixIcon: onToggleObscure == null
              ? null
              : IconButton(
                  icon: Icon(
                    obscure ? Icons.visibility_off : Icons.visibility,
                    color: Colors.white60,
                  ),
                  onPressed: onToggleObscure,
                ),
        ),
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({
    required this.color,
    required this.label,
    required this.icon,
  });
  final Color color;
  final String label;
  final IconData icon;
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 14),
          const SizedBox(width: 4),
          Text(label, style: TextStyle(color: color, fontSize: 12)),
        ],
      ),
    );
  }
}

class _ReadOnlyHint extends StatelessWidget {
  const _ReadOnlyHint(this.text);
  final String text;
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Text(
        text,
        style: const TextStyle(color: Colors.white60, fontSize: 12),
      ),
    );
  }
}

/// Server-LLM model picker. Renders the discovered model list as a dropdown
/// when the server advertised any, falls back to a free-form text field
/// otherwise so the user can type the id manually.
class _ServerModelPicker extends StatelessWidget {
  const _ServerModelPicker({
    required this.controller,
    required this.models,
    required this.loading,
    required this.error,
    required this.onRefresh,
  });

  final TextEditingController controller;
  final List<String> models;
  final bool loading;
  final String? error;
  final VoidCallback onRefresh;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Text(
                'Server LLM model',
                style: TextStyle(color: Colors.white60, fontSize: 12),
              ),
              const Spacer(),
              IconButton(
                onPressed: loading ? null : onRefresh,
                tooltip: 'Refresh model list',
                icon: loading
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white60,
                        ),
                      )
                    : const Icon(Icons.refresh,
                        color: Colors.white60, size: 18),
              ),
            ],
          ),
          if (models.isNotEmpty) ...[
            DropdownButtonFormField<String>(
              initialValue: models.contains(controller.text)
                  ? controller.text
                  : models.first,
              dropdownColor: const Color(0xFF1A1A1A),
              style: const TextStyle(color: Colors.white),
              decoration: const InputDecoration(
                enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.white24),
                ),
                focusedBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.white),
                ),
              ),
              items: [
                for (final m in models)
                  DropdownMenuItem<String>(
                    value: m,
                    child: Text(m,
                        style: const TextStyle(color: Colors.white)),
                  ),
              ],
              onChanged: (v) {
                if (v != null) controller.text = v;
              },
            ),
            const SizedBox(height: 8),
            Text(
              '${models.length} model${models.length == 1 ? '' : 's'} '
              'discovered from /v1/models',
              style: const TextStyle(color: Colors.white38, fontSize: 11),
            ),
          ] else ...[
            TextField(
              controller: controller,
              autocorrect: false,
              style: const TextStyle(color: Colors.white),
              decoration: const InputDecoration(
                hintText: AppSettings.defaultServerLlmModel,
                hintStyle: TextStyle(color: Colors.white30),
                enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.white24),
                ),
                focusedBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.white),
                ),
              ),
            ),
            if (error != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  error!,
                  style: const TextStyle(
                      color: Colors.redAccent, fontSize: 11),
                ),
              )
            else
              const Padding(
                padding: EdgeInsets.only(top: 8),
                child: Text(
                  'No models discovered yet. Tap refresh once the server is '
                  'reachable, or type a model id manually.',
                  style: TextStyle(color: Colors.white38, fontSize: 11),
                ),
              ),
          ],
        ],
      ),
    );
  }
}

/// Account row in Settings. Subscribes to the FirebaseAuth user stream so
/// signing out elsewhere (or a token revocation server-side) reflects here
/// immediately. Surfaces email / anonymous status and a Sign-Out action.
class _AccountTile extends StatefulWidget {
  const _AccountTile();

  @override
  State<_AccountTile> createState() => _AccountTileState();
}

class _AccountTileState extends State<_AccountTile> {
  late final StreamSubscription<User?> _sub;
  User? _user;
  bool _signingOut = false;

  @override
  void initState() {
    super.initState();
    _user = AuthService.instance.currentUser;
    _sub = AuthService.instance.userChanges.listen((u) {
      if (!mounted) return;
      setState(() => _user = u);
    });
  }

  @override
  void dispose() {
    _sub.cancel();
    super.dispose();
  }

  Future<void> _signOut() async {
    setState(() => _signingOut = true);
    try {
      await AuthService.instance.signOut();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text('Sign out failed: $e')));
    } finally {
      if (mounted) setState(() => _signingOut = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final user = _user;
    if (user == null) {
      return const ListTile(
        leading: Icon(Icons.person_off_outlined, color: Colors.white54),
        title: Text('Not signed in', style: TextStyle(color: Colors.white)),
      );
    }

    final isAnonymous = user.isAnonymous;
    final providers = user.providerData.map((p) => p.providerId).toList();
    final providerLabel = isAnonymous
        ? 'Anonymous session'
        : providers.isEmpty
            ? 'Signed in'
            : 'via ${providers.map(_providerLabel).join(", ")}';

    return ListTile(
      leading: Icon(
        isAnonymous ? Icons.person_outline : Icons.person,
        color: Colors.white,
      ),
      title: Text(
        isAnonymous ? 'Guest account' : (user.email ?? user.uid),
        style: const TextStyle(color: Colors.white),
      ),
      subtitle: Text(
        providerLabel,
        style: const TextStyle(color: Colors.white60, fontSize: 12),
      ),
      trailing: _signingOut
          ? const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.white70,
              ),
            )
          : TextButton(
              onPressed: _signOut,
              child: const Text(
                'Sign out',
                style: TextStyle(color: Colors.redAccent),
              ),
            ),
    );
  }

  static String _providerLabel(String id) {
    switch (id) {
      case 'password':
        return 'email';
      case 'google.com':
        return 'Google';
      case 'apple.com':
        return 'Apple';
      case 'firebase':
        return 'Firebase';
      default:
        return id;
    }
  }
}
