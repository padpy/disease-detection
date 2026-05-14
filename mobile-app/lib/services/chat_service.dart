import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:gopher_eye/services/api_client.dart';
import 'package:gopher_eye/services/app_settings.dart';
import 'package:langchain/langchain.dart';
import 'package:langchain_openai/langchain_openai.dart';

/// Thrown when the chat is invoked without the active provider being properly
/// configured (missing OpenAI key, missing server URL, etc.). The chat screen
/// shows a settings prompt instead of bubbling the generic error to the user.
class ChatConfigException implements Exception {
  const ChatConfigException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Role of a turn in the in-memory chat transcript. Matches the OpenAI roles
/// 1:1 so we can pass through to either provider without remapping.
enum LlmRole { user, assistant, system }

/// One message in an in-memory chat session. Camera-driven chats don't
/// persist, so this is the smallest possible turn struct — no DB id, no
/// timestamps.
class LlmTurn {
  const LlmTurn({required this.role, required this.content});
  final LlmRole role;
  final String content;
}

/// Return shape for [ChatService.reply]. Single-string for now; kept as a
/// class (rather than a bare `String`) so we can grow the return without
/// touching every call site.
class ChatReply {
  const ChatReply({required this.content});
  final String content;
}

/// The four diagnoses the chatbot is constrained to. Matches the categories
/// in the system prompt so callers (chip handlers) can branch on the model's
/// pick when generating follow-up prompts.
enum LeafDiagnosis { healthy, downyMildew, powderyMildew, otherDisease }

/// System prompt for OpenAI follow-up turns in OpenAI-direct mode. The leaf
/// image is still attached on every follow-up here (unlike the local-LLM
/// path), so the model can re-examine it when the user asks for
/// explanations or treatments.
///
/// The symptom hints below are deliberate: without them the model tends to
/// confuse the two mildews (they're both "mildew" in name but look nothing
/// alike on the leaf). Anchoring on the top-of-leaf white powder vs. the
/// oily spots / underside fuzz keeps the classifications stable.
const String kAgronomistSystemPrompt =
    'You are an agronomist helping a grower follow up on a grape-leaf '
    'diagnosis. The leaf image is attached to each turn and the prior '
    'assistant turn contains the diagnosis. '
    'Use these symptom hints when discussing the leaf: '
    'powdery mildew shows a powdery white substance on the tops of leaves; '
    'downy mildew shows oily yellow, brown, or black spots on the top of the '
    'leaf, the same symptoms on the underside, or a white thicker mold on '
    'the underside. '
    'Respond in a concise, friendly tone. Do not change the diagnosis '
    'already given. Feel free to include relevant web links (e.g. extension '
    'program pages) as plain http(s) URLs when they help the grower.';

/// System prompt for the INITIAL diagnosis when OpenAI is the provider end
/// to end (i.e. no local model in the loop). OpenAI sees the leaf image and
/// must reply with the diagnosis line only — no greeting, no explanation —
/// so the bubble is uniform with the local-LLM-then-OpenAI path.
const String kOpenAiInitialDiagnosisPrompt =
    'You diagnose grape-leaf disease from a leaf image. Choose exactly one '
    'diagnosis: healthy, downy mildew, powdery mildew, or other disease. '
    'Use these symptom hints when deciding: '
    'powdery mildew shows a powdery white substance on the tops of leaves; '
    'downy mildew shows oily yellow, brown, or black spots on the top of the '
    'leaf, the same symptoms on the underside, or a white thicker mold on '
    'the underside. '
    'Respond with EXACTLY ONE line in the form '
    '"Diagnosis: <one of: healthy, downy mildew, powdery mildew, other disease>". '
    'No greeting, no acknowledgement, no explanation, no other text.';

/// First message we send on the user's behalf when the chat opens — paired
/// with the leaf crop, this elicits the initial diagnosis. Phrased as a
/// short instruction so the model sticks to the system-prompt format.
const String kInitialDiagnosisPrompt =
    'Please review the attached leaf image and provide your diagnosis.';

/// Prompt for the "Explain diagnosis" chip. Asks for a 3–4 sentence visual
/// breakdown of the symptoms, anchored to the diagnosis the model already
/// gave.
const String kExplainDiagnosisPrompt =
    'In 3 to 4 sentences, describe the visible symptoms on this leaf that '
    'support the diagnosis you just gave. Reference specific things you see '
    '(e.g. lesion shape, colour, distribution, texture) rather than general '
    'disease facts. Keep the tone friendly and concrete.';

/// System prompt for the local/server model during the initial diagnosis.
/// The specialist model is the only thing that ever sees the leaf image. We
/// keep this prompt deliberately narrow — small classifier-style models tend
/// to drift when asked to produce structured output, so we just ask for the
/// diagnosis in a couple of words. The OpenAI extraction step downstream
/// normalises whatever the model says into the canonical category + format.
const String kLocalDiagnosticPrompt =
    'You are a specialist grape-leaf disease model. Examine the attached '
    'leaf image and state the diagnosis. Choose one of: healthy, downy '
    'mildew, powdery mildew, or other disease. '
    'Symptom hints: powdery mildew shows a powdery white substance on the '
    'tops of leaves; downy mildew shows oily yellow, brown, or black '
    'spots on the top of the leaf, the same symptoms on the underside, '
    'or a white thicker mold on the underside. '
    'Respond with the diagnosis only — one short phrase, no preamble.';

/// System prompt for the OpenAI diagnosis-extraction step. Takes the local
/// specialist's terse reply and produces ONLY the canonical "Diagnosis: …"
/// line — no greeting, no commentary — so the bubble matches the OpenAI-
/// direct initial path. Run once per initial diagnosis (no chat history
/// attached). Temperature is forced to 0 at the call site so the mapping
/// is deterministic.
const String kDiagnosisExtractionPrompt =
    'A specialist grape-leaf disease model examined a leaf image and '
    'produced a short free-form diagnosis (provided by the user). '
    'Map the specialist\'s diagnosis to exactly one of: healthy, downy '
    'mildew, powdery mildew, or other disease. If the specialist\'s text '
    'doesn\'t clearly match any category, choose "other disease". '
    'Respond with EXACTLY ONE line in the form '
    '"Diagnosis: <one of: healthy, downy mildew, powdery mildew, other disease>". '
    'No greeting, no acknowledgement, no explanation, no other text.';

/// System prompt for OpenAI when it is being used as the follow-up brain
/// after a local-LLM diagnosis. OpenAI never sees the leaf image in this
/// mode; the prior assistant turn (the extracted diagnosis) is the only
/// thing it has to anchor on. Do not let it second-guess the diagnosis.
const String kFollowUpSystemPrompt =
    'You are an agronomist helping a grower follow up on a leaf diagnosis '
    'that a specialist colleague already produced. The diagnosis appears '
    'in the prior assistant turn — treat it as ground truth and do not '
    'change it. You do not have access to the leaf image and you do not '
    'have the specialist\'s underlying symptom notes, so frame any '
    'symptom-level claims as "typical for this diagnosis" rather than '
    '"observed on this leaf". '
    'When the user asks for explanations, treatments, or resources, '
    'answer in a concise, friendly tone. Feel free to include relevant '
    'web links (e.g. extension program pages) as plain http(s) URLs when '
    'they help the grower.';

/// Routes chat prompts to whichever LLM provider the user has configured. The
/// system prompt + chat history are built the same way regardless of provider
/// — only the underlying transport differs.
class ChatService {
  ChatService._();
  static final ChatService instance = ChatService._();

  /// Build the prompt for the "Help Resources for treatment" chip. Names the
  /// state-specific extension service when [state] is provided, otherwise
  /// defaults to the University of Minnesota Extension content as the user
  /// requested. The model is instructed to list common treatments first then
  /// point to the extension program(s).
  String treatmentResourcesPrompt({String? state}) {
    final trimmedState = state?.trim();
    final hasState = trimmedState != null && trimmedState.isNotEmpty;
    final extensionDirective = hasState
        ? 'Then point the user to $trimmedState\'s land-grant '
            'university extension program for grape leaf disease '
            '(or the closest neighbouring state\'s if $trimmedState '
            'doesn\'t have a strong viticulture extension). Name the '
            'program and, if you can recall it, the URL or search query '
            'they should use to find current bulletins.'
        : 'Then direct the user to the University of Minnesota Extension '
            'content on grape leaf disease as the default reference, '
            'naming the program and pointing them to umn.edu Extension '
            'pages on grape diseases.';
    return 'Based on the diagnosis you just gave for this leaf, list the '
        'common treatments growers use (cultural practices, fungicide '
        'classes, timing). Keep the list short — 3 to 5 bullets. '
        '$extensionDirective '
        'Close with a one-line reminder to confirm any chemical applications '
        'against current state labels.';
  }

  /// Send [userMessage] alongside [imagePng] for the active chatbot session.
  /// [history] is every prior turn in order, *not* including the user turn
  /// being sent now. Returns the assistant's reply as a [ChatReply]; the
  /// caller stores `content` in chat history and renders it in the bubble.
  ///
  /// Routing rules when the active provider is the local/server model:
  ///   * [initialDiagnosis] requests do a two-step chain — the local LLM
  ///     classifies the image into a short free-form diagnosis, then
  ///     OpenAI extracts that into the canonical category + user-facing
  ///     reply. The OpenAI-extracted text is what the user sees and what
  ///     goes into chat history for any follow-ups.
  ///   * Every other turn is routed to OpenAI, which never receives the
  ///     image — the canonical diagnosis in chat history is what it
  ///     anchors on.
  /// When the active provider is OpenAI, every turn goes to OpenAI with the
  /// leaf image attached — that flow is unchanged.
  Future<ChatReply> reply({
    required Uint8List imagePng,
    required List<LlmTurn> history,
    required String userMessage,
    bool initialDiagnosis = false,
  }) async {
    final provider = await AppSettings.getLlmProvider();
    final started = DateTime.now();

    if (provider == LlmProvider.server) {
      if (initialDiagnosis) {
        debugPrint(
          '[chat_service] thinking: local LLM (initial diagnosis), '
          'image=${imagePng.length}B, history=${history.length} turn(s)',
        );
        final localStart = DateTime.now();
        final rawLocal = await _replyServer(
          systemPrompt: kLocalDiagnosticPrompt,
          history: history,
          userMessage: userMessage,
          imagePng: imagePng,
        );
        debugPrint(
          '[chat_service] local LLM returned in '
          '${DateTime.now().difference(localStart).inMilliseconds}ms, '
          'raw=${rawLocal.length} chars; extracting via OpenAI…',
        );
        final extractStart = DateTime.now();
        final extracted = await _extractDiagnosisViaOpenAI(rawLocal);
        debugPrint(
          '[chat_service] done: chain finished in '
          '${DateTime.now().difference(started).inMilliseconds}ms '
          '(extract step ${DateTime.now().difference(extractStart).inMilliseconds}ms), '
          'extracted=${extracted.length} chars',
        );
        return ChatReply(content: extracted);
      }
      debugPrint(
        '[chat_service] thinking: OpenAI follow-up (server provider, '
        'no image), history=${history.length} turn(s)',
      );
      final raw = await _replyOpenAI(
        systemPrompt: kFollowUpSystemPrompt,
        history: history,
        userMessage: userMessage,
        imagePng: Uint8List(0),
      );
      debugPrint(
        '[chat_service] done: OpenAI follow-up in '
        '${DateTime.now().difference(started).inMilliseconds}ms, '
        'reply=${raw.length} chars',
      );
      return ChatReply(content: raw);
    }

    final systemPrompt = initialDiagnosis
        ? kOpenAiInitialDiagnosisPrompt
        : kAgronomistSystemPrompt;
    debugPrint(
      '[chat_service] thinking: OpenAI (full provider, image=${imagePng.length}B, '
      '${initialDiagnosis ? "initial — diagnosis-only" : "follow-up"}), '
      'history=${history.length} turn(s)',
    );
    final raw = await _replyOpenAI(
      systemPrompt: systemPrompt,
      history: history,
      userMessage: userMessage,
      imagePng: imagePng,
    );
    debugPrint(
      '[chat_service] done: OpenAI in '
      '${DateTime.now().difference(started).inMilliseconds}ms, '
      'reply=${raw.length} chars',
    );
    return ChatReply(content: raw);
  }

  /// One-shot OpenAI call that turns the local specialist's terse diagnosis
  /// (`rawLocalResponse`) into the canonical user-facing reply ending with
  /// "Diagnosis: …". No chat history attached — this is pure extraction.
  /// Temperature is 0 so the category mapping is deterministic.
  Future<String> _extractDiagnosisViaOpenAI(String rawLocalResponse) async {
    final apiKey = await AppSettings.getOpenAiApiKey();
    if (apiKey == null || apiKey.isEmpty) {
      throw const ChatConfigException(
        'OpenAI API key not set. Open settings to add one — it\'s needed '
        'to extract the diagnosis from the local model\'s reply.',
      );
    }
    final modelName = await AppSettings.getOpenAiModel();
    final llm = ChatOpenAI(
      apiKey: apiKey,
      defaultOptions: ChatOpenAIOptions(
        model: modelName,
        temperature: 0.0,
      ),
    );
    final messages = <ChatMessage>[
      ChatMessage.system(kDiagnosisExtractionPrompt),
      ChatMessage.humanText('Specialist\'s diagnosis:\n$rawLocalResponse'),
    ];
    final result = await llm.invoke(PromptValue.chat(messages));
    return result.output.content;
  }

  Future<String> _replyOpenAI({
    required String systemPrompt,
    required List<LlmTurn> history,
    required String userMessage,
    required Uint8List imagePng,
  }) async {
    final apiKey = await AppSettings.getOpenAiApiKey();
    if (apiKey == null || apiKey.isEmpty) {
      throw const ChatConfigException(
        'OpenAI API key not set. Open settings to add one.',
      );
    }
    final modelName = await AppSettings.getOpenAiModel();
    final llm = ChatOpenAI(
      apiKey: apiKey,
      defaultOptions: ChatOpenAIOptions(
        model: modelName,
        temperature: 0.4,
      ),
    );
    final messages = <ChatMessage>[
      ChatMessage.system(systemPrompt),
      for (final turn in history) _toLangchain(turn),
      _humanWithImage(userMessage, imagePng),
    ];
    final result = await llm.invoke(PromptValue.chat(messages));
    return result.output.content;
  }

  /// langchain_openai's mapper wraps the data in a `data:<mime>;base64,...`
  /// URI itself, so pass only the raw base64 string here — passing a full
  /// data URI double-wraps it and OpenAI rejects with "Invalid base64 image_url".
  ChatMessage _humanWithImage(String text, Uint8List imagePng) {
    if (imagePng.isEmpty) return ChatMessage.humanText(text);
    return ChatMessage.human(
      ChatMessageContent.multiModal([
        ChatMessageContent.text(text),
        ChatMessageContent.image(
          data: base64Encode(imagePng),
          mimeType: 'image/png',
        ),
      ]),
    );
  }

  Future<String> _replyServer({
    required String systemPrompt,
    required List<LlmTurn> history,
    required String userMessage,
    required Uint8List imagePng,
  }) async {
    final url = await AppSettings.getServerUrl();
    if (url == null) {
      throw const ChatConfigException(
        'Server URL not set. Open settings to point at a server.',
      );
    }
    final model = await AppSettings.getServerLlmModel();
    final messages = <Map<String, dynamic>>[
      {'role': 'system', 'content': systemPrompt},
      for (final turn in history)
        {'role': turn.role.name, 'content': turn.content},
      {'role': 'user', 'content': userMessage},
    ];
    try {
      return await ApiClient.instance.chatCompletion(
        messages: messages,
        model: model,
        imagePng: imagePng,
      );
    } on ApiNotConfiguredException catch (e) {
      throw ChatConfigException(e.toString());
    } on ApiException catch (e) {
      throw ChatConfigException('Server LLM error: ${e.message}');
    }
  }

  ChatMessage _toLangchain(LlmTurn turn) {
    switch (turn.role) {
      case LlmRole.user:
        return ChatMessage.humanText(turn.content);
      case LlmRole.assistant:
        return ChatMessage.ai(turn.content);
      case LlmRole.system:
        return ChatMessage.system(turn.content);
    }
  }
}

/// Parse [reply]'s last "Diagnosis: …" line into one of the four canonical
/// diagnoses. Returns null if no recognisable line is present (e.g. before
/// the first turn). Used by the chatbot UI to decide whether to surface the
/// follow-up chip row.
LeafDiagnosis? parseLeafDiagnosis(String reply) {
  final match = RegExp(
    r'diagnosis\s*:\s*([a-zA-Z ]+)',
    caseSensitive: false,
  ).firstMatch(reply);
  if (match == null) return null;
  final value = match.group(1)?.trim().toLowerCase() ?? '';
  if (value.startsWith('healthy')) return LeafDiagnosis.healthy;
  if (value.startsWith('downy')) return LeafDiagnosis.downyMildew;
  if (value.startsWith('powdery')) return LeafDiagnosis.powderyMildew;
  if (value.startsWith('other')) return LeafDiagnosis.otherDisease;
  return null;
}
