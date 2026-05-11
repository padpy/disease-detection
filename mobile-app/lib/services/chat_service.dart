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

/// Routes chat prompts to whichever LLM provider the user has configured. The
/// system prompt + chat history are built the same way regardless of provider
/// — only the underlying transport differs.
class ChatService {
  ChatService._();
  static final ChatService instance = ChatService._();

  /// Send [userMessage] alongside [imagePng] for the active chatbot session.
  /// [history] is every prior turn in order, *not* including the user turn
  /// being sent now. Returns the assistant's text reply.
  Future<String> reply({
    required Uint8List imagePng,
    required List<LlmTurn> history,
    required String userMessage,
  }) async {
    final provider = await AppSettings.getLlmProvider();
    const systemPrompt =
        'Identify the grape leaf disease and describe the key visual signs '
        'in the image.';

    switch (provider) {
      case LlmProvider.openai:
        return _replyOpenAI(
          systemPrompt: systemPrompt,
          history: history,
          userMessage: userMessage,
          imagePng: imagePng,
        );
      case LlmProvider.server:
        return _replyServer(
          systemPrompt: systemPrompt,
          history: history,
          userMessage: userMessage,
          imagePng: imagePng,
        );
    }
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
