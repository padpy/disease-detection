import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:gopher_eye/services/fhb_pipeline.dart';
import 'package:gopher_eye/services/wheat_head_pipeline.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

enum _OverlayMode { off, segments, disease }

/// Manual HSV-threshold validation tool. Picks an image from the photo roll,
/// runs the standard wheat-head segmentation pipeline (YOLO26 → SAM), then
/// lets the user sweep the 8 HSV bounds used by [FhbAnalyzer] and watch the
/// disease overlay update live. Nothing is persisted — the screen exists to
/// dial in `FhbThresholds` defaults against real captures.
class ValidationScreen extends StatefulWidget {
  const ValidationScreen({super.key});

  @override
  State<ValidationScreen> createState() => _ValidationScreenState();
}

class _ValidationScreenState extends State<ValidationScreen> {
  bool _picking = false;
  bool _busy = false;
  bool _recomputing = false;
  String? _error;

  // Pipeline outputs cached so threshold tweaks only re-run the FHB analyzer.
  Uint8List? _workingPng;
  img.Image? _workingImage;
  int _workingW = 0;
  int _workingH = 0;
  List<WheatHeadDetection> _detections = const [];
  Uint8List? _segmentationOverlayPng;
  Uint8List? _diseaseOverlayPng;

  // Aggregate stats from the most recent threshold pass.
  int _totalGreen = 0;
  int _totalNecro = 0;
  int _totalOther = 0;
  int _totalMask = 0;

  // Mutable copies of FhbThresholds.defaults driven by the sliders.
  int _greenHueMin = FhbThresholds.defaults.greenHueMin;
  int _greenHueMax = FhbThresholds.defaults.greenHueMax;
  int _greenSatMin = FhbThresholds.defaults.greenSatMin;
  int _greenValMin = FhbThresholds.defaults.greenValMin;
  int _necroHueMin = FhbThresholds.defaults.necroHueMin;
  int _necroHueMax = FhbThresholds.defaults.necroHueMax;
  int _necroSatMin = FhbThresholds.defaults.necroSatMin;
  int _necroValMin = FhbThresholds.defaults.necroValMin;

  _OverlayMode _overlayMode = _OverlayMode.disease;
  bool _greenExpanded = true;
  bool _necroExpanded = true;

  Timer? _debounce;

  FhbThresholds _currentThresholds() => FhbThresholds(
        greenHueMin: _greenHueMin,
        greenHueMax: _greenHueMax,
        greenSatMin: _greenSatMin,
        greenValMin: _greenValMin,
        necroHueMin: _necroHueMin,
        necroHueMax: _necroHueMax,
        necroSatMin: _necroSatMin,
        necroValMin: _necroValMin,
      );

  @override
  void dispose() {
    _debounce?.cancel();
    super.dispose();
  }

  Future<void> _pickAndProcess() async {
    if (_picking || _busy) return;
    _picking = true;
    try {
      final picker = ImagePicker();
      final picked = await picker.pickImage(source: ImageSource.gallery);
      if (picked == null) return;
      setState(() {
        _busy = true;
        _error = null;
        _workingPng = null;
        _workingImage = null;
        _detections = const [];
        _segmentationOverlayPng = null;
        _diseaseOverlayPng = null;
        _totalGreen = 0;
        _totalNecro = 0;
        _totalOther = 0;
        _totalMask = 0;
      });
      final result = await WheatHeadPipeline.instance.run(File(picked.path));
      final decoded =
          WheatHeadPipeline.instance.decodeWorkingImage(result.imagePng);
      if (!mounted) return;
      setState(() {
        _workingPng = result.imagePng;
        _workingImage = decoded;
        _workingW = result.width;
        _workingH = result.height;
        _detections = result.detections;
        _segmentationOverlayPng = result.overlayPng;
      });
      await _recomputeDisease();
    } catch (e, st) {
      debugPrint('[validation] pipeline failed: $e\n$st');
      if (mounted) setState(() => _error = '$e');
    } finally {
      _picking = false;
      if (mounted) setState(() => _busy = false);
    }
  }

  void _scheduleRecompute() {
    _debounce?.cancel();
    _debounce = Timer(const Duration(milliseconds: 80), _recomputeDisease);
  }

  Future<void> _recomputeDisease() async {
    final image = _workingImage;
    if (image == null || _detections.isEmpty) return;
    if (_recomputing) {
      // Coalesce: drop this call, the in-flight one will see the latest
      // threshold values when it finishes since it reads them per-iteration.
      return;
    }
    _recomputing = true;
    try {
      final thresholds = _currentThresholds();
      final pipe = WheatHeadPipeline.instance;
      final reports = <FhbReport>[];
      int g = 0, n = 0, o = 0, t = 0;
      for (final d in _detections) {
        final report = FhbAnalyzer.instance.analyze(
          workingImage: image,
          spikeMask: d.mask,
          maskWidth: _workingW,
          maskHeight: _workingH,
          bbox: d.bbox,
          thresholds: thresholds,
        );
        reports.add(report);
        g += report.greenCount;
        n += report.necroticCount;
        o += report.otherCount;
        t += report.totalPixels;
      }
      final overlay = pipe.renderCombinedDiseaseOverlay(
        width: _workingW,
        height: _workingH,
        reports: reports,
      );
      if (!mounted) return;
      setState(() {
        _diseaseOverlayPng = overlay;
        _totalGreen = g;
        _totalNecro = n;
        _totalOther = o;
        _totalMask = t;
      });
    } finally {
      _recomputing = false;
    }
  }

  void _resetThresholds() {
    setState(() {
      _greenHueMin = FhbThresholds.defaults.greenHueMin;
      _greenHueMax = FhbThresholds.defaults.greenHueMax;
      _greenSatMin = FhbThresholds.defaults.greenSatMin;
      _greenValMin = FhbThresholds.defaults.greenValMin;
      _necroHueMin = FhbThresholds.defaults.necroHueMin;
      _necroHueMax = FhbThresholds.defaults.necroHueMax;
      _necroSatMin = FhbThresholds.defaults.necroSatMin;
      _necroValMin = FhbThresholds.defaults.necroValMin;
    });
    _scheduleRecompute();
  }

  @override
  Widget build(BuildContext context) {
    final ratio = (_totalGreen + _totalNecro) == 0
        ? 0.0
        : _totalNecro / (_totalGreen + _totalNecro);
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: const Text(
          'FHB threshold validation',
          style: TextStyle(color: Colors.white),
        ),
        actions: [
          IconButton(
            tooltip: 'Reset thresholds',
            onPressed: _detections.isEmpty ? null : _resetThresholds,
            icon: const Icon(Icons.restore, color: Colors.white),
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(vertical: 16),
          children: [
            _buildPreview(),
            if (_detections.isNotEmpty) ...[
              const SizedBox(height: 12),
              _buildToggle(),
              const SizedBox(height: 12),
              _buildStats(ratio),
              const SizedBox(height: 8),
              _buildThresholdGroup(
                title: 'Healthy (green) HSV band',
                accent: const Color(0xFF00C853),
                expanded: _greenExpanded,
                onToggle: () =>
                    setState(() => _greenExpanded = !_greenExpanded),
                hueMin: _greenHueMin,
                hueMax: _greenHueMax,
                satMin: _greenSatMin,
                valMin: _greenValMin,
                onHueMin: (v) {
                  setState(() => _greenHueMin = v);
                  _scheduleRecompute();
                },
                onHueMax: (v) {
                  setState(() => _greenHueMax = v);
                  _scheduleRecompute();
                },
                onSatMin: (v) {
                  setState(() => _greenSatMin = v);
                  _scheduleRecompute();
                },
                onValMin: (v) {
                  setState(() => _greenValMin = v);
                  _scheduleRecompute();
                },
              ),
              _buildThresholdGroup(
                title: 'Necrotic (FHB) HSV band',
                accent: const Color(0xFFDC1E1E),
                expanded: _necroExpanded,
                onToggle: () =>
                    setState(() => _necroExpanded = !_necroExpanded),
                hueMin: _necroHueMin,
                hueMax: _necroHueMax,
                satMin: _necroSatMin,
                valMin: _necroValMin,
                onHueMin: (v) {
                  setState(() => _necroHueMin = v);
                  _scheduleRecompute();
                },
                onHueMax: (v) {
                  setState(() => _necroHueMax = v);
                  _scheduleRecompute();
                },
                onSatMin: (v) {
                  setState(() => _necroSatMin = v);
                  _scheduleRecompute();
                },
                onValMin: (v) {
                  setState(() => _necroValMin = v);
                  _scheduleRecompute();
                },
              ),
              const SizedBox(height: 24),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildPreview() {
    if (_busy) {
      return const SizedBox(
        height: 280,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(color: Colors.white),
              SizedBox(height: 12),
              Text(
                'Running segmentation pipeline…',
                style: TextStyle(color: Colors.white70),
              ),
            ],
          ),
        ),
      );
    }
    final png = _workingPng;
    if (png == null) {
      return Padding(
        padding: const EdgeInsets.all(16),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: const Color(0xFF1A1A1A),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.white12),
          ),
          child: Column(
            children: [
              const Icon(Icons.tune, color: Colors.white60, size: 48),
              const SizedBox(height: 12),
              const Text(
                'Validate FHB HSV thresholds against a captured image. Pick a '
                'photo from your library, run the wheat-head segmentation '
                'pipeline, then sweep the HSV bands until the red-tinted '
                'pixels match the lesions you can see.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: _pickAndProcess,
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('Load image from library'),
              ),
              if (_error != null) ...[
                const SizedBox(height: 12),
                Text(
                  _error!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      color: Colors.redAccent, fontSize: 12),
                ),
              ],
            ],
          ),
        ),
      );
    }
    final Uint8List? overlay = switch (_overlayMode) {
      _OverlayMode.off => null,
      _OverlayMode.segments => _segmentationOverlayPng,
      _OverlayMode.disease => _diseaseOverlayPng,
    };
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: AspectRatio(
              aspectRatio: _workingW / _workingH,
              child: InteractiveViewer(
                minScale: 1,
                maxScale: 6,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.memory(png,
                        gaplessPlayback: true, fit: BoxFit.contain),
                    if (overlay != null)
                      Image.memory(overlay,
                          gaplessPlayback: true, fit: BoxFit.contain),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: Text(
                  '${_detections.length} spike${_detections.length == 1 ? '' : 's'} detected',
                  style:
                      const TextStyle(color: Colors.white60, fontSize: 12),
                ),
              ),
              TextButton.icon(
                onPressed: _pickAndProcess,
                icon: const Icon(Icons.refresh,
                    size: 16, color: Colors.white),
                label: const Text(
                  'New image',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildToggle() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: SegmentedButton<_OverlayMode>(
        showSelectedIcon: false,
        segments: const [
          ButtonSegment(
            value: _OverlayMode.off,
            label: Text('Off'),
            icon: Icon(Icons.visibility_off_outlined),
          ),
          ButtonSegment(
            value: _OverlayMode.segments,
            label: Text('Segments'),
            icon: Icon(Icons.gesture),
          ),
          ButtonSegment(
            value: _OverlayMode.disease,
            label: Text('Disease'),
            icon: Icon(Icons.local_florist),
          ),
        ],
        selected: {_overlayMode},
        onSelectionChanged: (s) => setState(() => _overlayMode = s.first),
        style: ButtonStyle(
          foregroundColor: WidgetStateProperty.resolveWith(
            (s) => s.contains(WidgetState.selected)
                ? Colors.black
                : Colors.white70,
          ),
          backgroundColor: WidgetStateProperty.resolveWith(
            (s) => s.contains(WidgetState.selected)
                ? Colors.white
                : Colors.white10,
          ),
          side: WidgetStateProperty.all(
            const BorderSide(color: Colors.white24),
          ),
        ),
      ),
    );
  }

  Widget _buildStats(double ratio) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFF1A1A1A),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.white12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'AGGREGATE ACROSS ALL SPIKES',
              style: TextStyle(
                color: Colors.white60,
                fontSize: 11,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.6,
              ),
            ),
            const SizedBox(height: 8),
            _statRow('FHB ratio', '${(ratio * 100).toStringAsFixed(1)}%'),
            _statRow('Healthy (green) px', _totalGreen.toString()),
            _statRow('Necrotic px', _totalNecro.toString()),
            _statRow('Other px', _totalOther.toString()),
            _statRow('Total mask px', _totalMask.toString()),
          ],
        ),
      ),
    );
  }

  Widget _statRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          SizedBox(
            width: 150,
            child: Text(
              label,
              style: const TextStyle(color: Colors.white54, fontSize: 12),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildThresholdGroup({
    required String title,
    required Color accent,
    required bool expanded,
    required VoidCallback onToggle,
    required int hueMin,
    required int hueMax,
    required int satMin,
    required int valMin,
    required ValueChanged<int> onHueMin,
    required ValueChanged<int> onHueMax,
    required ValueChanged<int> onSatMin,
    required ValueChanged<int> onValMin,
  }) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFF1A1A1A),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.white12),
        ),
        clipBehavior: Clip.antiAlias,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            InkWell(
              onTap: onToggle,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 12, 8, 12),
                child: Row(
                  children: [
                    Container(
                      width: 10,
                      height: 10,
                      decoration: BoxDecoration(
                        color: accent,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        title,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    _BandSummary(
                      hueMin: hueMin,
                      hueMax: hueMax,
                      satMin: satMin,
                      valMin: valMin,
                    ),
                    const SizedBox(width: 8),
                    AnimatedRotation(
                      turns: expanded ? 0.5 : 0.0,
                      duration: const Duration(milliseconds: 180),
                      child: const Icon(
                        Icons.keyboard_arrow_down,
                        color: Colors.white60,
                        size: 20,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            AnimatedSize(
              duration: const Duration(milliseconds: 180),
              alignment: Alignment.topCenter,
              curve: Curves.easeOut,
              child: expanded
                  ? Padding(
                      padding: const EdgeInsets.fromLTRB(12, 0, 12, 6),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          _BandPreview(
                            hueMin: hueMin,
                            hueMax: hueMax,
                            satMin: satMin,
                            valMin: valMin,
                          ),
                          const SizedBox(height: 8),
                          _slider('Hue min', hueMin, 0, 179, onHueMin),
                          _slider('Hue max', hueMax, 0, 179, onHueMax),
                          _slider('Sat min', satMin, 0, 255, onSatMin),
                          _slider('Val min', valMin, 0, 255, onValMin),
                        ],
                      ),
                    )
                  : const SizedBox(width: double.infinity),
            ),
          ],
        ),
      ),
    );
  }

  Widget _slider(
    String label,
    int value,
    int min,
    int max,
    ValueChanged<int> onChanged,
  ) {
    return Row(
      children: [
        SizedBox(
          width: 64,
          child: Text(
            label,
            style: const TextStyle(color: Colors.white60, fontSize: 12),
          ),
        ),
        Expanded(
          child: Slider(
            value: value.toDouble().clamp(min.toDouble(), max.toDouble()),
            min: min.toDouble(),
            max: max.toDouble(),
            divisions: max - min,
            label: value.toString(),
            activeColor: Colors.white,
            inactiveColor: Colors.white24,
            onChanged: (v) => onChanged(v.round()),
          ),
        ),
        SizedBox(
          width: 36,
          child: Text(
            value.toString(),
            textAlign: TextAlign.right,
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
        ),
      ],
    );
  }
}

/// Compact "H 20–90 · S≥5 · V≥5" chip shown in collapsed band headers so the
/// user can see the current threshold values at a glance.
class _BandSummary extends StatelessWidget {
  const _BandSummary({
    required this.hueMin,
    required this.hueMax,
    required this.satMin,
    required this.valMin,
  });

  final int hueMin;
  final int hueMax;
  final int satMin;
  final int valMin;

  @override
  Widget build(BuildContext context) {
    return Text(
      'H $hueMin–$hueMax · S≥$satMin · V≥$valMin',
      style: const TextStyle(
        color: Colors.white54,
        fontSize: 11,
        fontFeatures: [FontFeature.tabularFigures()],
      ),
    );
  }
}

/// Visualizes an HSV band: a rainbow hue strip with the selected
/// [hueMin]..[hueMax] window highlighted, paired with a 2D saturation×value
/// swatch (at the band's mid-hue) showing the spread of in-band colors.
class _BandPreview extends StatelessWidget {
  const _BandPreview({
    required this.hueMin,
    required this.hueMax,
    required this.satMin,
    required this.valMin,
  });

  final int hueMin;
  final int hueMax;
  final int satMin;
  final int valMin;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: SizedBox(
            height: 22,
            child: CustomPaint(
              painter: _HueStripPainter(hueMin: hueMin, hueMax: hueMax),
            ),
          ),
        ),
        const SizedBox(height: 6),
        Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: SizedBox(
                width: 88,
                height: 56,
                child: CustomPaint(
                  painter: _SatValSwatchPainter(
                    hueMin: hueMin,
                    hueMax: hueMax,
                    satMin: satMin,
                    valMin: valMin,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),
            const Expanded(
              child: Text(
                'Rainbow shows the selected hue window. Swatch shows in-band '
                'colors at the band\'s mid-hue, swept across S≥min and V≥min.',
                style: TextStyle(color: Colors.white54, fontSize: 11),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

/// Renders a 0..179 hue rainbow at full S/V, dimming the out-of-band portion
/// and drawing white edge markers at [hueMin] and [hueMax]+1.
class _HueStripPainter extends CustomPainter {
  _HueStripPainter({required this.hueMin, required this.hueMax});

  final int hueMin;
  final int hueMax;

  @override
  void paint(Canvas canvas, Size size) {
    const steps = 180;
    final stepW = size.width / steps;
    for (int hue = 0; hue < steps; hue++) {
      final inBand = hue >= hueMin && hue <= hueMax;
      // OpenCV hue 0..179 maps to 0..358° in Flutter's HSVColor.
      final color = HSVColor.fromAHSV(
        1,
        hue * 2.0,
        inBand ? 1.0 : 0.45,
        inBand ? 1.0 : 0.35,
      ).toColor();
      canvas.drawRect(
        Rect.fromLTWH(hue * stepW, 0, stepW + 0.5, size.height),
        Paint()..color = color,
      );
    }
    final marker = Paint()
      ..color = Colors.white
      ..strokeWidth = 1.5;
    final left = (hueMin.clamp(0, 179)) * stepW;
    final right = ((hueMax.clamp(0, 179)) + 1) * stepW;
    canvas.drawLine(Offset(left, 0), Offset(left, size.height), marker);
    canvas.drawLine(Offset(right, 0), Offset(right, size.height), marker);
  }

  @override
  bool shouldRepaint(covariant _HueStripPainter old) =>
      old.hueMin != hueMin || old.hueMax != hueMax;
}

/// Renders a 2D swatch at the band's mid-hue: saturation increases left→right
/// from [satMin]..255, value increases bottom→top from [valMin]..255.
class _SatValSwatchPainter extends CustomPainter {
  _SatValSwatchPainter({
    required this.hueMin,
    required this.hueMax,
    required this.satMin,
    required this.valMin,
  });

  final int hueMin;
  final int hueMax;
  final int satMin;
  final int valMin;

  @override
  void paint(Canvas canvas, Size size) {
    final hueDeg = ((hueMin + hueMax) / 2.0) * 2.0;
    const cells = 16;
    final cellW = size.width / cells;
    final cellH = size.height / cells;
    for (int xi = 0; xi < cells; xi++) {
      final sRaw = satMin + (255 - satMin) * (xi / (cells - 1));
      final s = (sRaw / 255).clamp(0.0, 1.0);
      for (int yi = 0; yi < cells; yi++) {
        final vRaw = valMin + (255 - valMin) * ((cells - 1 - yi) / (cells - 1));
        final v = (vRaw / 255).clamp(0.0, 1.0);
        final color = HSVColor.fromAHSV(1, hueDeg, s, v).toColor();
        canvas.drawRect(
          Rect.fromLTWH(xi * cellW, yi * cellH, cellW + 0.5, cellH + 0.5),
          Paint()..color = color,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant _SatValSwatchPainter old) =>
      old.hueMin != hueMin ||
      old.hueMax != hueMax ||
      old.satMin != satMin ||
      old.valMin != valMin;
}
