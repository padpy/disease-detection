import 'package:flutter/material.dart';
import 'package:gopher_eye/screens/camera_screen.dart';
import 'package:gopher_eye/screens/samples_screen.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  static const _swipeDuration = Duration(milliseconds: 280);
  static const _swipeCurve = Curves.easeOut;

  final PageController _controller = PageController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _goToSamples() {
    _controller.animateToPage(1, duration: _swipeDuration, curve: _swipeCurve);
  }

  void _goToCamera() {
    _controller.animateToPage(0, duration: _swipeDuration, curve: _swipeCurve);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: PageView(
        controller: _controller,
        physics: const ClampingScrollPhysics(),
        children: [
          CameraScreen(onOpenSamples: _goToSamples),
          SamplesScreen(onBack: _goToCamera),
        ],
      ),
    );
  }
}
