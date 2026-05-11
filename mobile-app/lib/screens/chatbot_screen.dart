import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gopher_eye/services/chat_service.dart';
import 'package:gopher_eye/services/grape_leaf_pipeline.dart';

/// Ad-hoc chat about a single image. Holds the transcript in memory only —
/// the session ends when the user pops back to the camera. There's no
/// per-image persistence layer because chatbot mode is meant for transient
/// "ask about this leaf" conversations triggered from the camera.
class ChatbotScreen extends StatefulWidget {
  const ChatbotScreen({super.key, required this.imageBytes});

  /// The full prepared image (PNG bytes) the user just captured / picked.
  /// The screen runs YOLO leaf segmentation against this on mount, picks the
  /// most central leaf, and uses the cropped leaf for both the on-screen
  /// preview and the LLM payload.
  final Uint8List imageBytes;

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  final TextEditingController _input = TextEditingController();
  final ScrollController _scroll = ScrollController();
  final List<LlmTurn> _turns = [];
  bool _sending = false;
  bool _extracting = true;
  String? _error;

  /// PNG bytes of the most-central leaf. Falls back to [widget.imageBytes]
  /// when YOLO can't find a leaf so the chat still works on non-leaf shots.
  Uint8List? _leafPng;

  /// True when the leaf actually came from YOLO; false when we fell back to
  /// the full frame. Drives the preview chip text.
  bool _leafFromYolo = false;

  /// The model's most recent diagnosis, parsed out of its `Diagnosis: …`
  /// line. Drives whether the follow-up chip row is visible.
  LeafDiagnosis? _diagnosis;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  /// Run leaf extraction, then auto-send the initial diagnosis prompt so the
  /// user lands on a screen that already has the agronomist's read.
  Future<void> _bootstrap() async {
    try {
      final crop = await GrapeLeafPipeline.instance
          .findCentralLeafCrop(widget.imageBytes);
      if (!mounted) return;
      setState(() {
        _leafPng = crop?.pngBytes ?? widget.imageBytes;
        _leafFromYolo = crop != null;
        _extracting = false;
      });
    } catch (e, st) {
      debugPrint('[chatbot] leaf extraction failed: $e\n$st');
      if (!mounted) return;
      setState(() {
        _leafPng = widget.imageBytes;
        _leafFromYolo = false;
        _extracting = false;
      });
    }
    await _sendUserMessage(kInitialDiagnosisPrompt, visibleText: null);
  }

  /// Send [content] as a user turn. When [visibleText] is null the user turn
  /// is suppressed from the transcript (used for the auto-sent initial
  /// diagnosis prompt so the chat opens with the assistant's read, not a
  /// duplicated request from the user).
  Future<void> _sendUserMessage(
    String content, {
    String? visibleText,
  }) async {
    if (_sending) return;
    setState(() {
      _sending = true;
      _error = null;
      if (visibleText != null) {
        _turns.add(LlmTurn(role: LlmRole.user, content: visibleText));
      }
    });
    _scrollToBottom();
    try {
      // History is everything *before* this new user message — strip the
      // visible turn we just added so we don't double-send it.
      final history = visibleText == null
          ? List<LlmTurn>.from(_turns)
          : (List<LlmTurn>.from(_turns)..removeLast());
      final reply = await ChatService.instance.reply(
        imagePng: _leafPng ?? widget.imageBytes,
        history: history,
        userMessage: content,
      );
      if (!mounted) return;
      final parsed = parseLeafDiagnosis(reply);
      setState(() {
        _turns.add(LlmTurn(role: LlmRole.assistant, content: reply));
        if (parsed != null) _diagnosis = parsed;
      });
      _scrollToBottom();
    } on ChatConfigException catch (e) {
      if (!mounted) return;
      setState(() => _error = e.message);
    } catch (e, st) {
      debugPrint('[chatbot] reply failed: $e\n$st');
      if (!mounted) return;
      setState(() => _error = 'Chat failed: $e');
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _sendTyped() async {
    final text = _input.text.trim();
    if (text.isEmpty) return;
    _input.clear();
    await _sendUserMessage(text, visibleText: text);
  }

  Future<void> _onExplain() async {
    await _sendUserMessage(
      kExplainDiagnosisPrompt,
      visibleText: 'Explain diagnosis',
    );
  }

  Future<void> _onTreatment() async {
    final state = await _pickState();
    if (!mounted) return;
    final prompt =
        ChatService.instance.treatmentResourcesPrompt(state: state);
    final visible = state == null
        ? 'Help resources for treatment'
        : 'Help resources for treatment · $state';
    await _sendUserMessage(prompt, visibleText: visible);
  }

  void _onDiagnoseAnother() {
    Navigator.of(context).pop();
  }

  /// Bottom-sheet picker for the user's state. Returns the chosen state name
  /// or null when they pick "Use default (University of Minnesota)" / dismiss.
  Future<String?> _pickState() async {
    return showModalBottomSheet<String?>(
      context: context,
      backgroundColor: const Color(0xFF1A1A1A),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (ctx) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Padding(
                padding: EdgeInsets.fromLTRB(20, 20, 20, 6),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'Pick your state',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
              const Padding(
                padding: EdgeInsets.fromLTRB(20, 0, 20, 12),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'We\'ll point you at that state\'s university '
                    'extension program. Skip to default to '
                    'University of Minnesota content.',
                    style: TextStyle(color: Colors.white60, fontSize: 12),
                  ),
                ),
              ),
              ListTile(
                leading: const Icon(Icons.school_outlined,
                    color: Colors.lightBlueAccent),
                title: const Text(
                  'Use default (University of Minnesota)',
                  style: TextStyle(color: Colors.white),
                ),
                onTap: () => Navigator.of(ctx).pop(null),
              ),
              const Divider(height: 1, color: Colors.white12),
              Flexible(
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: _kUsStates.length,
                  itemBuilder: (_, i) {
                    final s = _kUsStates[i];
                    return ListTile(
                      title: Text(
                        s,
                        style: const TextStyle(color: Colors.white),
                      ),
                      onTap: () => Navigator.of(ctx).pop(s),
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _copy(String text) async {
    await Clipboard.setData(ClipboardData(text: text));
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(const SnackBar(
        content: Text('Copied'),
        duration: Duration(seconds: 1),
        behavior: SnackBarBehavior.floating,
      ));
  }

  @override
  Widget build(BuildContext context) {
    final mediaHeight = MediaQuery.of(context).size.height;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: const Text(
          'Chatbot',
          style: TextStyle(color: Colors.white, fontSize: 15),
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            _LeafPreview(
              bytes: _leafPng ?? widget.imageBytes,
              maxHeight: mediaHeight / 3,
              extracting: _extracting,
              fromYolo: _leafFromYolo,
            ),
            Expanded(
              child: _turns.isEmpty
                  ? _EmptyState(extracting: _extracting)
                  : ListView.builder(
                      controller: _scroll,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 12),
                      itemCount: _turns.length,
                      itemBuilder: (_, i) {
                        final turn = _turns[i];
                        return _Bubble(
                          turn: turn,
                          onLongPress: () => _copy(turn.content),
                        );
                      },
                    ),
            ),
            if (_diagnosis != null && !_extracting)
              _QuickActions(
                disabled: _sending,
                onExplain: _onExplain,
                onTreatment: _onTreatment,
                onDiagnoseAnother: _onDiagnoseAnother,
              ),
            if (_error != null)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                    horizontal: 16, vertical: 10),
                color: Colors.redAccent.withValues(alpha: 0.15),
                child: Row(
                  children: [
                    const Icon(Icons.error_outline,
                        color: Colors.redAccent, size: 18),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _error!,
                        style: const TextStyle(
                            color: Colors.redAccent, fontSize: 12),
                      ),
                    ),
                  ],
                ),
              ),
            _Composer(
              controller: _input,
              onSend: _sendTyped,
              sending: _sending,
            ),
          ],
        ),
      ),
    );
  }
}

/// Top-of-chat preview of the cropped leaf. Capped at 1/3 of screen height
/// per the product spec; the cap is applied as a max-height ConstrainedBox so
/// the leaf can render smaller (preserving aspect) on landscape devices.
class _LeafPreview extends StatelessWidget {
  const _LeafPreview({
    required this.bytes,
    required this.maxHeight,
    required this.extracting,
    required this.fromYolo,
  });

  final Uint8List bytes;
  final double maxHeight;
  final bool extracting;
  final bool fromYolo;

  @override
  Widget build(BuildContext context) {
    return ConstrainedBox(
      constraints: BoxConstraints(maxHeight: maxHeight),
      child: Container(
        width: double.infinity,
        color: const Color(0xFF111111),
        padding: const EdgeInsets.all(8),
        child: Stack(
          fit: StackFit.expand,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.memory(
                bytes,
                fit: BoxFit.contain,
                gaplessPlayback: true,
              ),
            ),
            if (extracting)
              const Positioned(
                left: 12,
                bottom: 10,
                child: _PreviewBadge(
                  icon: Icons.center_focus_strong,
                  label: 'Finding central leaf…',
                ),
              )
            else
              Positioned(
                left: 12,
                bottom: 10,
                child: _PreviewBadge(
                  icon: fromYolo
                      ? Icons.center_focus_strong
                      : Icons.image_outlined,
                  label: fromYolo
                      ? 'Central leaf'
                      : 'Full image (no leaf detected)',
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _PreviewBadge extends StatelessWidget {
  const _PreviewBadge({required this.icon, required this.label});
  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white24, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: Colors.lightBlueAccent),
          const SizedBox(width: 6),
          Text(
            label,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.turn, this.onLongPress});

  final LlmTurn turn;
  final VoidCallback? onLongPress;

  @override
  Widget build(BuildContext context) {
    final isUser = turn.role == LlmRole.user;
    final bg = isUser ? Colors.white : const Color(0xFF1E1E1E);
    final fg = isUser ? Colors.black : Colors.white;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: GestureDetector(
        onLongPress: onLongPress,
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 4),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width * 0.78,
          ),
          decoration: BoxDecoration(
            color: bg,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Text(
            turn.content,
            style: TextStyle(color: fg, fontSize: 14, height: 1.35),
          ),
        ),
      ),
    );
  }
}

/// Row of pressable "chat bubble" chips below the transcript. Surfaced once
/// the assistant has produced a parseable `Diagnosis: …` line so the user
/// has obvious follow-ups instead of an empty composer.
class _QuickActions extends StatelessWidget {
  const _QuickActions({
    required this.disabled,
    required this.onExplain,
    required this.onTreatment,
    required this.onDiagnoseAnother,
  });

  final bool disabled;
  final VoidCallback onExplain;
  final VoidCallback onTreatment;
  final VoidCallback onDiagnoseAnother;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [
          _ActionChip(
            label: 'Explain diagnosis',
            icon: Icons.auto_awesome_outlined,
            onTap: disabled ? null : onExplain,
          ),
          _ActionChip(
            label: 'Help resources for treatment',
            icon: Icons.medical_services_outlined,
            onTap: disabled ? null : onTreatment,
          ),
          _ActionChip(
            label: 'Diagnose another leaf',
            icon: Icons.camera_alt_outlined,
            onTap: disabled ? null : onDiagnoseAnother,
          ),
        ],
      ),
    );
  }
}

class _ActionChip extends StatelessWidget {
  const _ActionChip({
    required this.label,
    required this.icon,
    required this.onTap,
  });

  final String label;
  final IconData icon;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null;
    return GestureDetector(
      onTap: onTap,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 150),
        opacity: enabled ? 1.0 : 0.45,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: const Color(0xFF1E1E1E),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: Colors.lightBlueAccent, width: 1),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 14, color: Colors.lightBlueAccent),
              const SizedBox(width: 6),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.controller,
    required this.onSend,
    required this.sending,
  });

  final TextEditingController controller;
  final VoidCallback onSend;
  final bool sending;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      decoration: const BoxDecoration(
        color: Colors.black,
        border: Border(top: BorderSide(color: Colors.white12)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: TextField(
              controller: controller,
              minLines: 1,
              maxLines: 5,
              enabled: !sending,
              textCapitalization: TextCapitalization.sentences,
              style: const TextStyle(color: Colors.white, fontSize: 14),
              decoration: InputDecoration(
                hintText: sending ? 'Thinking…' : 'Ask about this leaf…',
                hintStyle: const TextStyle(color: Colors.white38),
                filled: true,
                fillColor: const Color(0xFF1A1A1A),
                contentPadding: const EdgeInsets.symmetric(
                    horizontal: 14, vertical: 10),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(20),
                  borderSide: BorderSide.none,
                ),
              ),
              onSubmitted: (_) => onSend(),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 44,
            height: 44,
            child: ElevatedButton(
              onPressed: sending ? null : onSend,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: Colors.black,
                shape: const CircleBorder(),
                padding: EdgeInsets.zero,
              ),
              child: sending
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.black,
                      ),
                    )
                  : const Icon(Icons.send, size: 20),
            ),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.extracting});
  final bool extracting;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.smart_toy_outlined,
                color: Colors.white24, size: 64),
            const SizedBox(height: 12),
            Text(
              extracting
                  ? 'Locating the central leaf…'
                  : 'Asking the agronomist for a diagnosis…',
              style: const TextStyle(color: Colors.white70, fontSize: 14),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            const SizedBox(
              width: 22,
              height: 22,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.white38,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Static list of US states + DC for the treatment-resources state picker.
/// Order is alphabetical so users can scan quickly. Kept module-private — the
/// chatbot is the only consumer.
const List<String> _kUsStates = [
  'Alabama',
  'Alaska',
  'Arizona',
  'Arkansas',
  'California',
  'Colorado',
  'Connecticut',
  'Delaware',
  'District of Columbia',
  'Florida',
  'Georgia',
  'Hawaii',
  'Idaho',
  'Illinois',
  'Indiana',
  'Iowa',
  'Kansas',
  'Kentucky',
  'Louisiana',
  'Maine',
  'Maryland',
  'Massachusetts',
  'Michigan',
  'Minnesota',
  'Mississippi',
  'Missouri',
  'Montana',
  'Nebraska',
  'Nevada',
  'New Hampshire',
  'New Jersey',
  'New Mexico',
  'New York',
  'North Carolina',
  'North Dakota',
  'Ohio',
  'Oklahoma',
  'Oregon',
  'Pennsylvania',
  'Rhode Island',
  'South Carolina',
  'South Dakota',
  'Tennessee',
  'Texas',
  'Utah',
  'Vermont',
  'Virginia',
  'Washington',
  'West Virginia',
  'Wisconsin',
  'Wyoming',
];
