import 'dart:math' as math;

import 'package:flutter/material.dart';

/// Animated Siri-style shimmer that paints a sweeping multicolor border on
/// top of [child]. The camera screen wraps its preview in this widget and
/// flips [active] on while "chatbot mode" is engaged so the user has a
/// constant, ambient signal that captures will route into a chat instead of
/// the normal sample-save pipeline.
class SiriShimmer extends StatefulWidget {
  const SiriShimmer({
    super.key,
    required this.active,
    required this.child,
  });

  final bool active;
  final Widget child;

  @override
  State<SiriShimmer> createState() => _SiriShimmerState();
}

class _SiriShimmerState extends State<SiriShimmer>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2400),
    );
    if (widget.active) _ctrl.repeat();
  }

  @override
  void didUpdateWidget(covariant SiriShimmer old) {
    super.didUpdateWidget(old);
    if (widget.active && !_ctrl.isAnimating) {
      _ctrl.repeat();
    } else if (!widget.active && _ctrl.isAnimating) {
      _ctrl.stop();
    }
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.passthrough,
      children: [
        widget.child,
        if (widget.active)
          Positioned.fill(
            child: IgnorePointer(
              child: AnimatedBuilder(
                animation: _ctrl,
                builder: (_, __) => CustomPaint(
                  painter: _SiriBorderPainter(progress: _ctrl.value),
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class _SiriBorderPainter extends CustomPainter {
  _SiriBorderPainter({required this.progress});

  final double progress;

  static const _hues = <Color>[
    Color(0xFFB388FF),
    Color(0xFF00E5FF),
    Color(0xFFFFEB3B),
    Color(0xFFFF4081),
    Color(0xFFB388FF),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final rotation = progress * 2 * math.pi;

    // Three blurred passes form a feathered halo around the edge.
    for (var i = 0; i < 3; i++) {
      final stroke = 14.0 + i * 8;
      final blur = 22.0 + i * 14;
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = stroke
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, blur)
        ..shader = SweepGradient(
          colors: _hues,
          transform: GradientRotation(rotation),
        ).createShader(rect);
      canvas.drawRect(rect.deflate(stroke / 2), paint);
    }

    // Crisp counter-rotating inner stroke traces the edge.
    final inner = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5
      ..shader = SweepGradient(
        colors: _hues,
        transform: GradientRotation(-rotation),
      ).createShader(rect);
    canvas.drawRect(rect.deflate(1.25), inner);
  }

  @override
  bool shouldRepaint(covariant _SiriBorderPainter old) =>
      old.progress != progress;
}
