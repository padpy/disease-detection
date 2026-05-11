import 'package:gopher_eye/services/app_settings.dart';

/// Thin compatibility shim — the underlying store is now [AppSettings]. Kept
/// so existing call sites (chat screen, chat service) don't need to change
/// when the broader settings menu was added.
class OpenAISettings {
  OpenAISettings._();

  static const defaultModel = AppSettings.defaultOpenAiModel;

  static Future<String?> getApiKey() => AppSettings.getOpenAiApiKey();
  static Future<void> setApiKey(String? key) =>
      AppSettings.setOpenAiApiKey(key);
  static bool get isApiKeyFromBuild => AppSettings.isOpenAiApiKeyFromBuild;
  static Future<String> getModel() => AppSettings.getOpenAiModel();
  static Future<void> setModel(String model) =>
      AppSettings.setOpenAiModel(model);
}
