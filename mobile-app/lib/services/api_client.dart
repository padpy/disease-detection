import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:gopher_eye/model/zerotier_config.dart';
import 'package:gopher_eye/services/app_settings.dart';
import 'package:gopher_eye/services/vpn_service.dart';
import 'package:http/http.dart' as http;

/// Thrown when the server URL is missing/blank in settings, so the caller can
/// route the user to the settings screen instead of bubbling a TCP error.
class ApiNotConfiguredException implements Exception {
  const ApiNotConfiguredException();
  @override
  String toString() =>
      'Server URL is not configured. Open settings to add one.';
}

/// Generic transport / status-code error wrapping a structured response from
/// the Flask backend ({"error": {"message": …, "code": …}}).
class ApiException implements Exception {
  ApiException(this.statusCode, this.message, [this.body]);
  final int statusCode;
  final String message;
  final String? body;

  @override
  String toString() => 'ApiException($statusCode): $message';
}

/// HTTP wrapper for the Flask backend. All endpoints resolve the server URL
/// from [AppSettings] at call-time so URL changes take effect immediately
/// without rebuilding the singleton.
class ApiClient {
  ApiClient._();
  static final ApiClient instance = ApiClient._();

  final http.Client _client = http.Client();

  /// Default per-request timeout. Overridable per call.
  static const Duration _defaultTimeout = Duration(seconds: 30);

  Future<Uri> _uri(String path, [Map<String, String?>? query]) async {
    final base = await AppSettings.getServerUrl();
    if (base == null) throw const ApiNotConfiguredException();
    // If the user enabled the VPN but the tunnel hasn't come up yet (cold
    // start, in-flight reconnect, etc.), wait briefly so the call routes
    // through it. After the timeout we fall through and let the request go
    // out plaintext — the user can always disable VPN if it's broken.
    if (await AppSettings.getVpnEnabled() &&
        VpnService.instance.current != VpnState.connected) {
      await VpnService.instance.waitUntilReady(
          timeout: const Duration(seconds: 5));
    }
    final cleanedQuery = <String, String>{};
    if (query != null) {
      for (final entry in query.entries) {
        if (entry.value == null) continue;
        cleanedQuery[entry.key] = entry.value!;
      }
    }
    final root = Uri.parse(base);
    return root.replace(
      path: path,
      queryParameters: cleanedQuery.isEmpty ? null : cleanedQuery,
    );
  }

  Map<String, String> _jsonHeaders() => const {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };

  Never _raise(http.Response res) {
    String message = 'HTTP ${res.statusCode}';
    try {
      final decoded = jsonDecode(res.body);
      if (decoded is Map && decoded['error'] is Map) {
        final err = decoded['error'] as Map;
        if (err['message'] is String) message = err['message'] as String;
      }
    } catch (_) {}
    throw ApiException(res.statusCode, message, res.body);
  }

  // ---------------------------------------------------------------------
  // Health / connectivity
  // ---------------------------------------------------------------------

  /// Simple GET /status round-trip; returns true on 200, false on any failure.
  Future<bool> ping({Duration timeout = const Duration(seconds: 5)}) async {
    try {
      final uri = await _uri('/status');
      final res = await _client.get(uri).timeout(timeout);
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ---------------------------------------------------------------------
  // Samples
  // ---------------------------------------------------------------------

  Future<List<Map<String, dynamic>>> listSamples({
    String? user,
    String? detectionMode,
    int? since,
    int? limit,
  }) async {
    final uri = await _uri('/samples', {
      'user': user,
      'detection_mode': detectionMode,
      if (since != null) 'since': '$since',
      if (limit != null) 'limit': '$limit',
    });
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode != 200) _raise(res);
    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    return List<Map<String, dynamic>>.from(decoded['samples'] as List);
  }

  Future<Map<String, dynamic>> createSample({
    required Uint8List imageBytes,
    required String filename,
    required Map<String, dynamic> metadata,
  }) async {
    final uri = await _uri('/samples');
    final req = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes(
        'image',
        imageBytes,
        filename: filename,
      ))
      ..fields['data'] = jsonEncode(metadata);
    final streamed = await req.send().timeout(const Duration(seconds: 60));
    final res = await http.Response.fromStream(streamed);
    if (res.statusCode != 201 && res.statusCode != 200) _raise(res);
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>?> getSample(int sampleId,
      {List<String> includeBlobs = const []}) async {
    final uri = await _uri('/samples/$sampleId', {
      if (includeBlobs.isNotEmpty) 'include_blobs': includeBlobs.join(','),
    });
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode == 404) return null;
    if (res.statusCode != 200) _raise(res);
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> updateSample(
      int sampleId, Map<String, dynamic> partial) async {
    final uri = await _uri('/samples/$sampleId');
    final res = await _client
        .patch(uri, headers: _jsonHeaders(), body: jsonEncode(partial))
        .timeout(_defaultTimeout);
    if (res.statusCode != 200) _raise(res);
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<void> deleteSample(int sampleId) async {
    final uri = await _uri('/samples/$sampleId');
    final res = await _client.delete(uri).timeout(_defaultTimeout);
    if (res.statusCode != 204 && res.statusCode != 200) _raise(res);
  }

  Future<Uint8List> getSampleSource(int sampleId) async {
    final uri = await _uri('/samples/$sampleId/source');
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode != 200) _raise(res);
    return res.bodyBytes;
  }

  Future<Uint8List?> getSampleBlob(int sampleId, String kind) async {
    final uri = await _uri('/samples/$sampleId/blob/$kind');
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode == 404) return null;
    if (res.statusCode != 200) _raise(res);
    return res.bodyBytes;
  }

  Future<Map<String, dynamic>> putSampleBlob(
    int sampleId,
    String kind,
    Uint8List png, {
    int? width,
    int? height,
  }) async {
    final uri = await _uri('/samples/$sampleId/blob/$kind', {
      if (width != null) 'width': '$width',
      if (height != null) 'height': '$height',
    });
    final res = await _client
        .put(uri, headers: const {'Content-Type': 'image/png'}, body: png)
        .timeout(const Duration(seconds: 60));
    if (res.statusCode != 200) _raise(res);
    if (res.body.isEmpty) return const <String, dynamic>{};
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  // ---------------------------------------------------------------------
  // Instances
  // ---------------------------------------------------------------------

  Future<List<Map<String, dynamic>>> listInstances(int sampleId,
      {List<String> includeBlobs = const []}) async {
    final uri = await _uri('/samples/$sampleId/instances', {
      if (includeBlobs.isNotEmpty) 'include_blobs': includeBlobs.join(','),
    });
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode != 200) _raise(res);
    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    return List<Map<String, dynamic>>.from(decoded['instances'] as List);
  }

  /// Replace every instance for the sample atomically. The server expects
  /// each instance map to carry base64-encoded ``mask_png``/``preview_png``
  /// fields plus bbox/centroid/score/etc.
  Future<List<Map<String, dynamic>>> replaceInstances(
    int sampleId,
    List<Map<String, dynamic>> payloads,
  ) async {
    final uri = await _uri('/samples/$sampleId/instances');
    final res = await _client
        .put(
          uri,
          headers: _jsonHeaders(),
          body: jsonEncode({'instances': payloads}),
        )
        .timeout(const Duration(seconds: 90));
    if (res.statusCode != 200) _raise(res);
    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    return List<Map<String, dynamic>>.from(decoded['instances'] as List);
  }

  Future<Uint8List?> getInstanceBlob(int instanceId, String kind) async {
    final uri = await _uri('/instances/$instanceId/blob/$kind');
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode == 404) return null;
    if (res.statusCode != 200) _raise(res);
    return res.bodyBytes;
  }

  // ---------------------------------------------------------------------
  // Detection (legacy /dl/segmentation* + /plant/* polling)
  // ---------------------------------------------------------------------

  /// Start a remote detection job. ``task`` is "leaf" (grape) or "spike"
  /// (wheat FHB). Returns the server-issued ``plant_id``.
  Future<String> submitDetection({
    required Uint8List imageBytes,
    required String filename,
    required String task,
    Map<String, dynamic>? data,
  }) async {
    final route =
        task == 'spike' ? '/dl/segmentation_spike' : '/dl/segmentation';
    final uri = await _uri(route);
    final req = http.MultipartRequest('PUT', uri)
      ..files.add(http.MultipartFile.fromBytes(
        'image',
        imageBytes,
        filename: filename,
      ));
    if (data != null && data.isNotEmpty) {
      req.fields['data'] = jsonEncode(data);
    }
    final streamed = await req.send().timeout(const Duration(seconds: 60));
    final res = await http.Response.fromStream(streamed);
    if (res.statusCode != 200) _raise(res);
    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    return decoded['plant_id'] as String;
  }

  Future<String> getPlantStatus(String plantId) async {
    final uri = await _uri('/plant/status', {'plant_id': plantId});
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode != 200) _raise(res);
    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    return (decoded['status'] as String?) ?? 'unknown';
  }

  Future<Map<String, dynamic>> getPlantData(String plantId) async {
    final uri = await _uri('/plant/data', {'plant_id': plantId});
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode != 200) _raise(res);
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  // ---------------------------------------------------------------------
  // Chat / LLM
  // ---------------------------------------------------------------------

  /// Returns the model ids advertised by ``/v1/models``. Empty list when the
  /// server hasn't enabled the chatbot. Callers should treat any ApiException
  /// as "no models available" so the settings UI stays usable.
  Future<List<String>> listLlmModels() async {
    final uri = await _uri('/v1/models');
    final res = await _client.get(uri).timeout(_defaultTimeout);
    if (res.statusCode != 200) _raise(res);
    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    final data = decoded['data'];
    if (data is! List) return const [];
    return [
      for (final entry in data)
        if (entry is Map && entry['id'] is String) entry['id'] as String,
    ];
  }

  /// POST /v1/chat/completions with an OpenAI-shaped messages payload. Pass
  /// [imagePng] to attach an image to the most recent user message — the
  /// server's BLIP_Qwen requires at least one image. Returns the assistant's
  /// text reply.
  Future<String> chatCompletion({
    required List<Map<String, dynamic>> messages,
    required String model,
    Uint8List? imagePng,
    int maxTokens = 256,
    double temperature = 0.4,
  }) async {
    final uri = await _uri('/v1/chat/completions');
    final payloadMessages = List<Map<String, dynamic>>.from(messages);
    if (imagePng != null && payloadMessages.isNotEmpty) {
      // Inject the image into the last user message so the server can pick
      // it up as the active image (its ``_messages_to_prompt_and_image``
      // walks messages and uses the most recent image_url part).
      final dataUri = 'data:image/png;base64,${base64Encode(imagePng)}';
      final attached = <Map<String, dynamic>>[];
      var injected = false;
      for (var i = payloadMessages.length - 1; i >= 0; i--) {
        final msg = Map<String, dynamic>.from(payloadMessages[i]);
        if (!injected && msg['role'] == 'user') {
          final existing = msg['content'];
          final parts = <Map<String, dynamic>>[
            if (existing is String && existing.isNotEmpty)
              {'type': 'text', 'text': existing}
            else if (existing is List)
              ...existing.cast<Map<String, dynamic>>(),
            {
              'type': 'image_url',
              'image_url': {'url': dataUri},
            },
          ];
          msg['content'] = parts;
          payloadMessages[i] = msg;
          injected = true;
        }
        attached.insert(0, payloadMessages[i]);
      }
    }

    final body = jsonEncode({
      'model': model,
      'messages': payloadMessages,
      'max_tokens': maxTokens,
      'temperature': temperature,
      'stream': false,
    });
    final res = await _client
        .post(uri, headers: _jsonHeaders(), body: body)
        .timeout(const Duration(seconds: 120));
    if (res.statusCode != 200) _raise(res);
    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    final choices = decoded['choices'];
    if (choices is! List || choices.isEmpty) {
      throw ApiException(200, 'no choices in chat response', res.body);
    }
    final first = choices.first as Map<String, dynamic>;
    final message = first['message'] as Map<String, dynamic>?;
    final content = message?['content'];
    if (content is! String) {
      throw ApiException(200, 'no message.content in chat response', res.body);
    }
    return content;
  }

  // ---------------------------------------------------------------------
  // ZeroTier config (broker peer issuance, proxied through the server)
  // ---------------------------------------------------------------------

  /// Register this device's ZeroTier node id with the server (which proxies
  /// to the broker) and return the resulting peer config. Idempotent: if the
  /// user is already registered, the server returns the existing assignment
  /// when the node id matches, or rotates the node id while keeping the
  /// assigned IP if it differs.
  Future<ZeroTierConfig> registerVpnConfig(String clientNodeId) async {
    final uri = await _uri('/vpn/config');
    final res = await _client
        .post(
          uri,
          headers: await _authJsonHeaders(),
          body: jsonEncode({'client_node_id': clientNodeId}),
        )
        .timeout(_defaultTimeout);
    if (res.statusCode != 200) _raise(res);
    return ZeroTierConfig.fromJson(
        jsonDecode(res.body) as Map<String, dynamic>);
  }

  /// Fetch the previously-issued config for this user. Returns null if the
  /// server has never registered a peer for this account.
  Future<ZeroTierConfig?> getVpnConfig() async {
    final uri = await _uri('/vpn/config');
    final res = await _client
        .get(uri, headers: await _authHeaders())
        .timeout(_defaultTimeout);
    if (res.statusCode == 404) return null;
    if (res.statusCode != 200) _raise(res);
    return ZeroTierConfig.fromJson(
        jsonDecode(res.body) as Map<String, dynamic>);
  }

  /// Revoke this user's ZeroTier peer. The reconciler deauthorizes it from
  /// the live network within seconds.
  Future<void> revokeVpnConfig() async {
    final uri = await _uri('/vpn/config');
    final res = await _client
        .delete(uri, headers: await _authHeaders())
        .timeout(_defaultTimeout);
    if (res.statusCode != 204 && res.statusCode != 200 && res.statusCode != 404) {
      _raise(res);
    }
  }

  Future<Map<String, String>> _authHeaders() async {
    final token = await AppSettings.getAuthToken();
    if (token == null) {
      throw ApiException(
        401,
        'No auth token configured. The Firebase Auth flow must populate '
        'AppSettings.setAuthToken before /vpn/config can be called.',
      );
    }
    return {
      'Authorization': 'Bearer $token',
      'Accept': 'application/json',
    };
  }

  Future<Map<String, String>> _authJsonHeaders() async {
    final headers = await _authHeaders();
    return {...headers, 'Content-Type': 'application/json'};
  }
}
