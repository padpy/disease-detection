import 'package:shared_preferences/shared_preferences.dart';

/// Where the on-device detection pipeline runs. ``local`` is the default and
/// keeps everything on the phone. ``remote`` uploads the capture to the
/// configured server and waits for it to return labels + masks.
enum DetectionLocation {
  local,
  remote;

  String get id => switch (this) {
        DetectionLocation.local => 'local',
        DetectionLocation.remote => 'remote',
      };

  String get label => switch (this) {
        DetectionLocation.local => 'On device',
        DetectionLocation.remote => 'On server',
      };

  static DetectionLocation fromId(String? id) {
    for (final v in DetectionLocation.values) {
      if (v.id == id) return v;
    }
    return DetectionLocation.local;
  }
}

/// Which LLM the chat screen routes prompts to. ``openai`` keeps the existing
/// LangChain ChatOpenAI path; ``server`` uses the configured backend's
/// ``/v1/chat/completions`` endpoint (OpenAI-compatible, multimodal).
enum LlmProvider {
  openai,
  server;

  String get id => switch (this) {
        LlmProvider.openai => 'openai',
        LlmProvider.server => 'server',
      };

  String get label => switch (this) {
        LlmProvider.openai => 'OpenAI',
        LlmProvider.server => 'Server LLM',
      };

  static LlmProvider fromId(String? id) {
    for (final v in LlmProvider.values) {
      if (v.id == id) return v;
    }
    return LlmProvider.openai;
  }
}

/// One source of truth for all user-configurable settings: server backend,
/// detection location, sync toggle, chat LLM provider + per-provider models.
/// Persists to SharedPreferences. Build-time defines (e.g.
/// ``--dart-define=OPENAI_API_KEY=…``) still win over UI values for keys.
class AppSettings {
  AppSettings._();

  // Keys
  static const _kServerUrl = 'app_server_url';
  static const _kDetectionLocation = 'app_detection_location';
  static const _kSyncEnabled = 'app_sync_enabled';
  static const _kLlmProvider = 'app_llm_provider';
  static const _kServerLlmModel = 'app_server_llm_model';
  static const _kOpenAiApiKey = 'openai_api_key';
  static const _kOpenAiModel = 'openai_model';
  static const _kUserName = 'app_user_name';
  static const _kVpnEnabled = 'app_vpn_enabled';
  static const _kZtNetworkId = 'app_zt_network_id';
  static const _kAuthToken = 'app_auth_token';

  static const _envOpenAiKey = String.fromEnvironment('OPENAI_API_KEY');
  static const _envServerUrl = String.fromEnvironment('GOPHER_EYE_SERVER_URL');

  static const defaultOpenAiModel = 'gpt-4o-mini';
  static const defaultServerLlmModel = 'gopher-eye-grape-leaf';
  static const defaultZtNetworkId = '6ab565387a1297b5';

  // ---------- Server ----------

  /// Base URL like ``http://192.168.1.42:5555``. No trailing slash. Returns
  /// null when nothing is configured.
  static Future<String?> getServerUrl() async {
    if (_envServerUrl.isNotEmpty) return _envServerUrl;
    final prefs = await SharedPreferences.getInstance();
    final value = prefs.getString(_kServerUrl);
    if (value == null || value.trim().isEmpty) return null;
    return _normalizeUrl(value);
  }

  static Future<void> setServerUrl(String? value) async {
    final prefs = await SharedPreferences.getInstance();
    if (value == null || value.trim().isEmpty) {
      await prefs.remove(_kServerUrl);
    } else {
      await prefs.setString(_kServerUrl, _normalizeUrl(value));
    }
  }

  static bool get isServerUrlFromBuild => _envServerUrl.isNotEmpty;

  static String _normalizeUrl(String raw) {
    var url = raw.trim();
    while (url.endsWith('/')) {
      url = url.substring(0, url.length - 1);
    }
    return url;
  }

  // ---------- Detection location ----------

  static Future<DetectionLocation> getDetectionLocation() async {
    final prefs = await SharedPreferences.getInstance();
    return DetectionLocation.fromId(prefs.getString(_kDetectionLocation));
  }

  static Future<void> setDetectionLocation(DetectionLocation value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kDetectionLocation, value.id);
  }

  // ---------- Backup/sync ----------

  /// When true, every locally-saved sample (and instances/blobs as detection
  /// completes) is mirrored to the configured server.
  static Future<bool> getSyncEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_kSyncEnabled) ?? false;
  }

  static Future<void> setSyncEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kSyncEnabled, value);
  }

  // ---------- User identifier (sample.user on server) ----------

  static Future<String> getUserName() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_kUserName) ?? '';
  }

  static Future<void> setUserName(String value) async {
    final prefs = await SharedPreferences.getInstance();
    if (value.trim().isEmpty) {
      await prefs.remove(_kUserName);
    } else {
      await prefs.setString(_kUserName, value.trim());
    }
  }

  // ---------- LLM provider ----------

  static Future<LlmProvider> getLlmProvider() async {
    final prefs = await SharedPreferences.getInstance();
    return LlmProvider.fromId(prefs.getString(_kLlmProvider));
  }

  static Future<void> setLlmProvider(LlmProvider value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kLlmProvider, value.id);
  }

  // ---------- OpenAI ----------

  static Future<String?> getOpenAiApiKey() async {
    if (_envOpenAiKey.isNotEmpty) return _envOpenAiKey;
    final prefs = await SharedPreferences.getInstance();
    final stored = prefs.getString(_kOpenAiApiKey);
    if (stored == null || stored.isEmpty) return null;
    return stored;
  }

  static Future<void> setOpenAiApiKey(String? key) async {
    final prefs = await SharedPreferences.getInstance();
    if (key == null || key.isEmpty) {
      await prefs.remove(_kOpenAiApiKey);
    } else {
      await prefs.setString(_kOpenAiApiKey, key);
    }
  }

  static bool get isOpenAiApiKeyFromBuild => _envOpenAiKey.isNotEmpty;

  static Future<String> getOpenAiModel() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_kOpenAiModel) ?? defaultOpenAiModel;
  }

  static Future<void> setOpenAiModel(String model) async {
    final prefs = await SharedPreferences.getInstance();
    if (model.isEmpty) {
      await prefs.remove(_kOpenAiModel);
    } else {
      await prefs.setString(_kOpenAiModel, model);
    }
  }

  // ---------- ZeroTier VPN toggle ----------

  /// When true, the app should bring the ZeroTier tunnel up at launch and
  /// keep it up while running. Persisted across launches so users do not
  /// have to re-enable after every cold start.
  static Future<bool> getVpnEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_kVpnEnabled) ?? false;
  }

  static Future<void> setVpnEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kVpnEnabled, value);
  }

  /// The 16-hex ZeroTier network this device is enrolled on. Returned by the
  /// broker via the server during enrollment and cached here so the Settings
  /// UI can display it even before the tunnel comes up. Falls back to
  /// [defaultZtNetworkId] when nothing has been persisted yet.
  static Future<String> getZtNetworkId() async {
    final prefs = await SharedPreferences.getInstance();
    final value = prefs.getString(_kZtNetworkId);
    if (value == null || value.isEmpty) return defaultZtNetworkId;
    return value;
  }

  /// Persist or clear the cached ZeroTier network id. Empty/null clears the
  /// entry. Validates 16 lowercase hex characters; throws [FormatException]
  /// for anything else (callers in the UI should pre-validate so this only
  /// fires on coding mistakes).
  static Future<void> setZtNetworkId(String? value) async {
    final prefs = await SharedPreferences.getInstance();
    final trimmed = value?.trim().toLowerCase();
    if (trimmed == null || trimmed.isEmpty) {
      await prefs.remove(_kZtNetworkId);
      return;
    }
    if (!RegExp(r'^[0-9a-f]{16}$').hasMatch(trimmed)) {
      throw const FormatException(
        'ZeroTier network id must be exactly 16 hex characters',
      );
    }
    await prefs.setString(_kZtNetworkId, trimmed);
  }

  // ---------- Bearer token for authenticated server endpoints ----------

  /// Bearer token attached as `Authorization: Bearer <value>` on routes that
  /// require auth (currently only `/vpn/config*`). Intended to hold a
  /// Firebase ID token once the sign-in flow is wired up; until then it can
  /// be populated manually for local testing.
  static Future<String?> getAuthToken() async {
    final prefs = await SharedPreferences.getInstance();
    final value = prefs.getString(_kAuthToken);
    if (value == null || value.isEmpty) return null;
    return value;
  }

  static Future<void> setAuthToken(String? value) async {
    final prefs = await SharedPreferences.getInstance();
    if (value == null || value.isEmpty) {
      await prefs.remove(_kAuthToken);
    } else {
      await prefs.setString(_kAuthToken, value);
    }
  }

  // ---------- Server-hosted LLM model ----------

  static Future<String> getServerLlmModel() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_kServerLlmModel) ?? defaultServerLlmModel;
  }

  static Future<void> setServerLlmModel(String model) async {
    final prefs = await SharedPreferences.getInstance();
    if (model.isEmpty) {
      await prefs.remove(_kServerLlmModel);
    } else {
      await prefs.setString(_kServerLlmModel, model);
    }
  }
}
