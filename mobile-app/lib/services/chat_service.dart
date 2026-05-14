import 'dart:convert';
import 'dart:typed_data';

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

/// The four diagnoses the chatbot is constrained to. Matches the categories
/// in the system prompt so callers (chip handlers) can branch on the model's
/// pick when generating follow-up prompts.
enum LeafDiagnosis { healthy, downyMildew, powderyMildew, otherDisease }

/// Standing system prompt that defines the chatbot's role + the strict
/// 4-category diagnosis vocabulary. Used as the system message on every
/// outbound chat request so follow-ups stay on-topic and on-vocabulary.
///
/// The symptom hints below are deliberate: without them the model tends to
/// confuse the two mildews (they're both "mildew" in name but look nothing
/// alike on the leaf). Anchoring on the top-of-leaf white powder vs. the
/// oily spots / underside fuzz keeps the classifications stable.
const String kAgronomistSystemPrompt =
    'You are an agronomist helping diagnose leaf disease. '
    'Choose if the leaf has one of the following diagnosis: '
    'healthy, downy mildew, powdery mildew, or other disease. '
    'Use these symptom hints when deciding: '
    'powdery mildew shows a powdery white substance on the tops of leaves; '
    'downy mildew shows oily yellow, brown, or black spots on the top of the '
    'leaf, the same symptoms on the underside, or a white thicker mold on '
    'the underside. '
    'Respond in a concise, friendly manner acknowledging reviewing the leaf '
    'and providing the diagnosis. Keep the initial reply to two short '
    'sentences and end with the chosen diagnosis on its own line in the form '
    '"Diagnosis: <one of: healthy, downy mildew, powdery mildew, other disease>".';

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
/// The specialist model is the only thing that ever sees the leaf image, so
/// it carries the full diagnostic vocabulary plus tone constraints — its
/// output is what the user sees verbatim and what OpenAI uses as the
/// descriptive context for any later follow-ups.
const String kLocalDiagnosticPrompt =
    'You are a specialist grape-leaf disease model. Examine the attached '
    'leaf image and list the visible symptoms you observe (colour, texture, '
    'location on leaf, distribution). Then choose one diagnosis from: '
    'healthy, downy mildew, powdery mildew, or other disease. '
    'Use these symptom hints: '
    'powdery mildew shows a powdery white substance on the tops of leaves; '
    'downy mildew shows oily yellow, brown, or black spots on the top of '
    'the leaf, the same symptoms on the underside, or a white thicker mold '
    'on the underside. '
    'Respond in a concise, friendly manner acknowledging reviewing the leaf '
    'and providing the diagnosis. End with the chosen diagnosis on its own '
    'line in the form '
    '"Diagnosis: <one of: healthy, downy mildew, powdery mildew, other disease>".';

/// System prompt for OpenAI when it is being used as the follow-up brain
/// after a local-LLM diagnosis. OpenAI never sees the leaf image in this
/// mode; the specialist's notes (passed via chat history) are the only
/// description of what the leaf looks like, so the prompt tells OpenAI to
/// treat those notes as ground truth.
const String kFollowUpSystemPrompt =
    'You are an agronomist helping a grower follow up on a leaf diagnosis '
    'that a specialist colleague already produced. You do not have access '
    'to the leaf image — rely entirely on the specialist\'s notes earlier '
    'in this conversation (observed symptoms and the chosen diagnosis) for '
    'what the leaf looks like. Do not change the specialist\'s diagnosis. '
    'When the user asks for explanations, treatments, or resources, ground '
    'every observation in those notes and answer in a concise, friendly '
    'tone. Feel free to include relevant web links (e.g. extension program '
    'pages) as plain http(s) URLs when they help the grower.';

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
  /// being sent now. Returns the assistant's text reply.
  ///
  /// Routing rules when the active provider is the local/server model:
  ///   * [initialDiagnosis] requests go to the local LLM and its output is
  ///     returned verbatim — the user only ever sees the specialist's
  ///     diagnosis on the first turn.
  ///   * Every other turn is routed to OpenAI, which never receives the
  ///     image. The chat history (containing the local LLM's symptoms +
  ///     diagnosis) is the descriptive context it works from.
  /// When the active provider is OpenAI, every turn goes to OpenAI with the
  /// leaf image attached — that flow is unchanged.
  Future<String> reply({
    required Uint8List imagePng,
    required List<LlmTurn> history,
    required String userMessage,
    bool initialDiagnosis = false,
  }) async {
    final provider = await AppSettings.getLlmProvider();

    if (provider == LlmProvider.server) {
      if (initialDiagnosis) {
        return _replyServer(
          systemPrompt: kLocalDiagnosticPrompt,
          history: history,
          userMessage: userMessage,
          imagePng: imagePng,
        );
      }
      return _replyOpenAI(
        systemPrompt: kFollowUpSystemPrompt,
        history: history,
        userMessage: userMessage,
        imagePng: Uint8List(0),
      );
    }

    return _replyOpenAI(
      systemPrompt: kAgronomistSystemPrompt,
      history: history,
      userMessage: userMessage,
      imagePng: imagePng,
    );
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
