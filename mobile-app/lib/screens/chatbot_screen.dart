import 'dart:async';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gopher_eye/services/chat_service.dart';
import 'package:gopher_eye/services/grape_leaf_pipeline.dart';
import 'package:url_launcher/url_launcher.dart';

/// Ad-hoc chat about a single image. Holds the transcript in memory only —
/// the session ends when the user pops back to the camera. There's no
/// per-image persistence layer because chatbot mode is meant for transient
/// "ask about this leaf" conversations triggered from the camera.
///
/// The flow is entirely chip-driven: leaf extraction + initial diagnosis run
/// automatically on mount, and follow-ups are restricted to the action chips
/// at the bottom of the screen. No free-form text input.
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

/// One step in the rotating thinking-status sequence. We cycle through the
/// list while waiting on the LLM (or on YOLO during the initial extract),
/// swapping the bubble text every [_kStatusRotateInterval].
const Duration _kStatusRotateInterval = Duration(milliseconds: 1600);

/// Statuses shown in the thinking bubble for each phase. Kept here (rather
/// than inlined at the call site) so copy lives in one place and the lists
/// can be tweaked without disturbing the state machine.
const List<String> _kStatusExtracting = [
  'Scanning the photo for leaves…',
  'Picking the most central leaf…',
  'Cropping the leaf for review…',
];
const List<String> _kStatusDiagnosing = [
  'Reviewing the leaf…',
  'Looking for mildew, lesions, and discolouration…',
  'Comparing against healthy / downy / powdery patterns…',
  'Forming a diagnosis…',
  'Summarising the findings…',
];
const List<String> _kStatusExplaining = [
  'Re-examining visible symptoms…',
  'Picking out the most telling signs…',
  'Drafting the explanation…',
];
const List<String> _kStatusTreatment = [
  'Gathering common treatments…',
  'Looking up extension program resources…',
  'Putting the recommendations together…',
];

class _ChatbotScreenState extends State<ChatbotScreen> {
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

  /// Active rotating-status list + which slot is currently visible. Empty
  /// when no work is in flight (the bubble is hidden in that case).
  List<String> _statuses = const [];
  int _statusIdx = 0;
  Timer? _statusTimer;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
    _scroll.dispose();
    super.dispose();
  }

  /// Run leaf extraction, then auto-send the initial diagnosis prompt so the
  /// user lands on a screen that already has the agronomist's read. The
  /// status bubble cycles through extract → diagnose copy so the user can
  /// see what stage we're on.
  Future<void> _bootstrap() async {
    _startStatuses(_kStatusExtracting);
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
    await _sendUserMessage(
      kInitialDiagnosisPrompt,
      visibleText: null,
      statuses: _kStatusDiagnosing,
      initialDiagnosis: true,
    );
  }

  /// Send [content] as a user turn. When [visibleText] is null the user turn
  /// is suppressed from the transcript (used for the auto-sent initial
  /// diagnosis prompt so the chat opens with the assistant's read, not a
  /// duplicated request from the user).
  ///
  /// [initialDiagnosis] flips on the local-then-OpenAI hybrid pipeline in
  /// [ChatService.reply] when the server LLM is the active provider.
  Future<void> _sendUserMessage(
    String content, {
    String? visibleText,
    required List<String> statuses,
    bool initialDiagnosis = false,
  }) async {
    if (_sending) return;
    debugPrint(
      '[chatbot] thinking start: '
      'initialDiagnosis=$initialDiagnosis, '
      'visibleText=${visibleText == null ? "(suppressed)" : '"$visibleText"'}, '
      'history=${_turns.length} turn(s)',
    );
    setState(() {
      _sending = true;
      _error = null;
      if (visibleText != null) {
        _turns.add(LlmTurn(role: LlmRole.user, content: visibleText));
      }
    });
    _startStatuses(statuses);
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
        initialDiagnosis: initialDiagnosis,
      );
      if (!mounted) return;
      final parsed = parseLeafDiagnosis(reply.content);
      debugPrint(
        '[chatbot] thinking done: reply=${reply.content.length} chars, '
        'diagnosis=${parsed?.name ?? "(unparsed)"}',
      );
      setState(() {
        _turns.add(LlmTurn(role: LlmRole.assistant, content: reply.content));
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
      if (mounted) {
        _stopStatuses();
        setState(() => _sending = false);
      }
    }
  }

  Future<void> _onExplain() async {
    await _sendUserMessage(
      kExplainDiagnosisPrompt,
      visibleText: 'Explain diagnosis',
      statuses: _kStatusExplaining,
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
    await _sendUserMessage(
      prompt,
      visibleText: visible,
      statuses: _kStatusTreatment,
    );
  }

  void _onDiagnoseAnother() {
    Navigator.of(context).pop();
  }

  /// Begin (or replace) the rotating-status bubble with [statuses]. Resets to
  /// the first entry and ticks forward every [_kStatusRotateInterval]. The
  /// last entry stays put once reached so we never wrap mid-thought.
  void _startStatuses(List<String> statuses) {
    _statusTimer?.cancel();
    if (statuses.isEmpty) {
      setState(() {
        _statuses = const [];
        _statusIdx = 0;
      });
      return;
    }
    debugPrint('[chatbot] thinking: ${statuses.first}');
    setState(() {
      _statuses = statuses;
      _statusIdx = 0;
    });
    _statusTimer = Timer.periodic(_kStatusRotateInterval, (_) {
      if (!mounted) return;
      if (_statusIdx >= _statuses.length - 1) return;
      setState(() => _statusIdx += 1);
      debugPrint('[chatbot] thinking: ${_statuses[_statusIdx]}');
    });
  }

  void _stopStatuses() {
    _statusTimer?.cancel();
    _statusTimer = null;
    debugPrint('[chatbot] thinking stopped');
    setState(() {
      _statuses = const [];
      _statusIdx = 0;
    });
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

  /// Items to render in the transcript: every committed turn, plus a
  /// synthetic "thinking" turn at the tail whenever the agronomist is
  /// working. Returning a single list keeps the ListView builder simple and
  /// lets the thinking bubble share styling with real assistant bubbles.
  List<_TranscriptItem> _buildItems() {
    final items = <_TranscriptItem>[
      for (final t in _turns) _TranscriptItem.turn(t),
    ];
    final showThinking = _extracting || _sending;
    if (showThinking && _statuses.isNotEmpty) {
      final idx = _statusIdx.clamp(0, _statuses.length - 1);
      items.add(_TranscriptItem.thinking(_statuses[idx]));
    }
    return items;
  }

  @override
  Widget build(BuildContext context) {
    final mediaHeight = MediaQuery.of(context).size.height;
    final items = _buildItems();
    final chipsDisabled = _sending || _extracting;
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
              child: items.isEmpty
                  ? const SizedBox.shrink()
                  : ListView.builder(
                      controller: _scroll,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 12),
                      itemCount: items.length,
                      itemBuilder: (_, i) {
                        final item = items[i];
                        if (item.thinkingStatus != null) {
                          return _ThinkingBubble(
                            status: item.thinkingStatus!,
                          );
                        }
                        final turn = item.turn!;
                        return _Bubble(
                          turn: turn,
                          onLongPress: () => _copy(turn.content),
                        );
                      },
                    ),
            ),
            if (_diagnosis != null)
              _QuickActions(
                disabled: chipsDisabled,
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
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }
}

/// Discriminated union for the transcript list. Exactly one of [turn] or
/// [thinkingStatus] is non-null. Saves us juggling two parallel lists in the
/// state object.
class _TranscriptItem {
  const _TranscriptItem._({this.turn, this.thinkingStatus});
  factory _TranscriptItem.turn(LlmTurn turn) =>
      _TranscriptItem._(turn: turn);
  factory _TranscriptItem.thinking(String status) =>
      _TranscriptItem._(thinkingStatus: status);

  final LlmTurn? turn;
  final String? thinkingStatus;
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

class _Bubble extends StatefulWidget {
  const _Bubble({required this.turn, this.onLongPress});

  final LlmTurn turn;
  final VoidCallback? onLongPress;

  @override
  State<_Bubble> createState() => _BubbleState();
}

class _BubbleState extends State<_Bubble> {
  /// Recognisers owned by this bubble. Each linkified URL needs its own
  /// recogniser; we hold them so we can dispose them when the bubble's
  /// content changes or the bubble unmounts.
  final List<TapGestureRecognizer> _recognisers = [];

  @override
  void didUpdateWidget(covariant _Bubble oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.turn.content != widget.turn.content) _disposeRecognisers();
  }

  @override
  void dispose() {
    _disposeRecognisers();
    super.dispose();
  }

  void _disposeRecognisers() {
    for (final r in _recognisers) {
      r.dispose();
    }
    _recognisers.clear();
  }

  @override
  Widget build(BuildContext context) {
    final isUser = widget.turn.role == LlmRole.user;
    final bg = isUser ? Colors.white : const Color(0xFF1E1E1E);
    final fg = isUser ? Colors.black : Colors.white;
    final linkColor =
        isUser ? Colors.blue.shade800 : Colors.lightBlueAccent;
    final base = TextStyle(color: fg, fontSize: 14, height: 1.35);

    _disposeRecognisers();
    final spans = _buildSpans(widget.turn.content, base, linkColor);

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: GestureDetector(
        onLongPress: widget.onLongPress,
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
          child: Text.rich(TextSpan(style: base, children: spans)),
        ),
      ),
    );
  }

  List<InlineSpan> _buildSpans(
    String content,
    TextStyle baseStyle,
    Color linkColor,
  ) {
    final spans = <InlineSpan>[];
    var cursor = 0;
    for (final match in _kUrlPattern.allMatches(content)) {
      if (match.start > cursor) {
        spans.add(TextSpan(text: content.substring(cursor, match.start)));
      }
      final raw = match.group(0)!;
      // Trailing punctuation often sits next to URLs in prose ("see foo.org.").
      // Strip it from the link target and push it back into the surrounding
      // text so the recogniser doesn't activate on the period.
      final stripped = _stripTrailingPunctuation(raw);
      final url = stripped.url;
      final recogniser = TapGestureRecognizer()
        ..onTap = () => _open(url);
      _recognisers.add(recogniser);
      spans.add(TextSpan(
        text: url,
        style: baseStyle.copyWith(
          color: linkColor,
          decoration: TextDecoration.underline,
          decorationColor: linkColor,
        ),
        recognizer: recogniser,
      ));
      if (stripped.trailing.isNotEmpty) {
        spans.add(TextSpan(text: stripped.trailing));
      }
      cursor = match.end;
    }
    if (cursor < content.length) {
      spans.add(TextSpan(text: content.substring(cursor)));
    }
    return spans;
  }

  Future<void> _open(String url) async {
    final uri = Uri.tryParse(url);
    if (uri == null) return;
    final ok = await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (!ok && mounted) {
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(
          content: Text('Could not open $url'),
          duration: const Duration(seconds: 2),
          behavior: SnackBarBehavior.floating,
        ));
    }
  }
}

/// Matches plain http(s) URLs. We deliberately don't try to detect bare
/// domains — the LLM is instructed to emit fully qualified links.
final RegExp _kUrlPattern = RegExp(
  r'https?://[^\s<>()\[\]{}"' "'" r']+',
  caseSensitive: false,
);

class _StrippedUrl {
  const _StrippedUrl(this.url, this.trailing);
  final String url;
  final String trailing;
}

_StrippedUrl _stripTrailingPunctuation(String raw) {
  const trailing = '.,;:!?)]}>';
  var end = raw.length;
  while (end > 0 && trailing.contains(raw[end - 1])) {
    end -= 1;
  }
  if (end == raw.length) return _StrippedUrl(raw, '');
  return _StrippedUrl(raw.substring(0, end), raw.substring(end));
}

/// Assistant-side "thinking" bubble. Shows the current rotating-status line
/// plus an animated three-dot indicator so the user can tell work is still
/// in flight even between status swaps.
class _ThinkingBubble extends StatelessWidget {
  const _ThinkingBubble({required this.status});
  final String status;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        decoration: BoxDecoration(
          color: const Color(0xFF1E1E1E),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.white12),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const _TypingDots(),
            const SizedBox(width: 10),
            Flexible(
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 280),
                transitionBuilder: (child, anim) =>
                    FadeTransition(opacity: anim, child: child),
                child: Text(
                  status,
                  key: ValueKey(status),
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 14,
                    height: 1.35,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Three-dot animated indicator that bounces with a staggered phase. Used in
/// the thinking bubble alongside the rotating status text.
class _TypingDots extends StatefulWidget {
  const _TypingDots();
  @override
  State<_TypingDots> createState() => _TypingDotsState();
}

class _TypingDotsState extends State<_TypingDots>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1100),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (_, __) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(3, (i) {
            final phase = (_controller.value - i * 0.18) % 1.0;
            final scale = 0.6 + 0.6 * (1 - (phase * 2 - 1).abs()).clamp(0, 1);
            return Padding(
              padding: EdgeInsets.only(right: i == 2 ? 0 : 4),
              child: Transform.scale(
                scale: scale,
                child: Container(
                  width: 6,
                  height: 6,
                  decoration: const BoxDecoration(
                    color: Colors.lightBlueAccent,
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            );
          }),
        );
      },
    );
  }
}

/// Row of pressable "chat bubble" chips below the transcript. Surfaced once
/// the assistant has produced a parseable `Diagnosis: …` line so the user
/// has obvious follow-ups — this is the only way the user can drive the
/// conversation forward (free-form text input is intentionally disabled).
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
