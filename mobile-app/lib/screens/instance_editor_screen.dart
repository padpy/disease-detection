import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:gopher_eye/model/sample_instance.dart';
import 'package:gopher_eye/services/sample_repository.dart';
import 'package:gopher_eye/services/wheat_head_pipeline.dart';

enum _EditorTool {
  /// Tap-to-prompt SAM with a foreground/background point. The currently
  /// shown mask is replaced by the SAM output.
  sam,

  /// Drag to paint mask pixels (1 = inside).
  brush,

  /// Drag to erase mask pixels (0 = outside).
  erase,
}

/// Per-instance editor: shows the cropped working image plus the binary mask
/// overlay, and lets the user refine the segmentation either by re-prompting
/// SAM or by direct brush/erase strokes.
///
/// The editor operates on the *full* working image (not the cropped tile)
/// so SAM coordinates remain consistent with the original detection. The
/// canvas is windowed to the instance's bbox plus padding via
/// [InteractiveViewer].
/// Sentinel returned by the editor when the user deletes the instance, so
/// the caller can distinguish that case from a no-op cancel.
class InstanceEditorResult {
  const InstanceEditorResult.saved(this.instance) : deleted = false;
  const InstanceEditorResult.deleted(this.instance) : deleted = true;
  final SampleInstance instance;
  final bool deleted;
}

class InstanceEditorScreen extends StatefulWidget {
  /// Edit an existing [instance].
  const InstanceEditorScreen({
    super.key,
    required SampleInstance this.instance,
    required this.workingPng,
    required this.workingWidth,
    required this.workingHeight,
  }) : sampleId = -1;

  /// Create a brand-new instance for [sampleId]. The editor opens with an
  /// empty mask spanning the full working image; the user paints / SAM-prompts
  /// to populate it, and Save inserts a new row.
  const InstanceEditorScreen.create({
    super.key,
    required this.sampleId,
    required this.workingPng,
    required this.workingWidth,
    required this.workingHeight,
  }) : instance = null;

  final SampleInstance? instance;
  final int sampleId;
  final Uint8List workingPng;
  final int workingWidth;
  final int workingHeight;

  bool get isCreating => instance == null;
  int get effectiveSampleId => instance?.sampleId ?? sampleId;

  @override
  State<InstanceEditorScreen> createState() => _InstanceEditorScreenState();
}

class _InstanceEditorScreenState extends State<InstanceEditorScreen> {
  late Uint8List _mask; // 1 = inside, 0 = outside; sized W×H
  late int _maskW;
  late int _maskH;

  ui.Image? _fullImage;
  final TransformationController _viewer = TransformationController();
  bool _viewerInitialised = false;
  EditableEmbedding? _embedding;
  bool _embeddingLoading = false;
  String? _embeddingError;

  _EditorTool _tool = _EditorTool.brush;
  double _brushRadius = 12; // in working-image pixels
  // SAM point label: 1 = foreground, 0 = background.
  int _samLabel = 1;
  bool _busy = false;
  bool _dirty = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    final pipe = WheatHeadPipeline.instance;
    final existing = widget.instance;
    if (existing != null) {
      final decoded = pipe.decodeMaskPng(existing.maskPng);
      _mask = decoded.mask;
      _maskW = decoded.width;
      _maskH = decoded.height;
    } else {
      // Brand-new instance: empty mask covering the full working image.
      _maskW = widget.workingWidth;
      _maskH = widget.workingHeight;
      _mask = Uint8List(_maskW * _maskH);
      // Default to SAM mode for new instances — that's what the user usually
      // wants when seeding a fresh mask.
      _tool = _EditorTool.sam;
    }

    _decodeFull();
  }

  Future<void> _decodeFull() async {
    final codec = await ui.instantiateImageCodec(widget.workingPng);
    try {
      final frame = await codec.getNextFrame();
      if (!mounted) {
        frame.image.dispose();
        return;
      }
      setState(() => _fullImage = frame.image);
    } finally {
      codec.dispose();
    }
  }

  Future<void> _ensureEmbedding() async {
    if (_embedding != null || _embeddingLoading) return;
    setState(() {
      _embeddingLoading = true;
      _embeddingError = null;
    });
    try {
      final emb = await WheatHeadPipeline.instance
          .prepareEditFromPng(widget.workingPng);
      if (!mounted) {
        emb.release();
        return;
      }
      setState(() {
        _embedding = emb;
        _embeddingLoading = false;
      });
    } catch (e, st) {
      debugPrint('[editor] embedding failed: $e\n$st');
      if (!mounted) return;
      setState(() {
        _embeddingError = '$e';
        _embeddingLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _fullImage?.dispose();
    _viewer.dispose();
    _embedding?.release();
    super.dispose();
  }

  void _paintAt(Offset working, {required bool inside}) {
    final value = inside ? 1 : 0;
    final r = _brushRadius;
    final r2 = r * r;
    final cx = working.dx;
    final cy = working.dy;
    final x0 = (cx - r).floor().clamp(0, _maskW - 1);
    final y0 = (cy - r).floor().clamp(0, _maskH - 1);
    final x1 = (cx + r).ceil().clamp(0, _maskW - 1);
    final y1 = (cy + r).ceil().clamp(0, _maskH - 1);
    for (int y = y0; y <= y1; y++) {
      final dy = y - cy;
      for (int x = x0; x <= x1; x++) {
        final dx = x - cx;
        if (dx * dx + dy * dy <= r2) {
          _mask[y * _maskW + x] = value;
        }
      }
    }
    _dirty = true;
  }

  Future<void> _runSamPoint(Offset working) async {
    if (_busy) return;
    await _ensureEmbedding();
    final emb = _embedding;
    if (emb == null) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      // Pass the original detection bbox along with the new point so SAM
      // refines the existing region instead of starting over. For freshly
      // created instances we don't have a bbox yet, so the point alone is
      // the only prompt.
      final newMask = await WheatHeadPipeline.instance.predict(
        embedding: emb,
        origW: _maskW,
        origH: _maskH,
        points: [working],
        pointLabels: [_samLabel],
        bbox: widget.instance?.bbox,
      );
      if (!mounted) return;
      setState(() {
        _mask = newMask;
        _busy = false;
        _dirty = true;
      });
    } catch (e, st) {
      debugPrint('[editor] sam predict failed: $e\n$st');
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _busy = false;
      });
    }
  }

  Future<void> _save() async {
    final existing = widget.instance;
    if (_busy) return;
    if (!_dirty) {
      // Nothing changed — close without round-tripping the DB.
      Navigator.of(context).pop();
      return;
    }
    final maskBbox = _bboxFromMask();
    if (maskBbox == null && widget.isCreating) {
      // Refusing to persist an empty instance.
      setState(() => _error = 'Mask is empty — paint or run SAM first.');
      return;
    }
    setState(() => _busy = true);
    try {
      final pipe = WheatHeadPipeline.instance;
      final maskPng = pipe.encodeMaskPng(_mask, _maskW, _maskH);
      final fallback = existing?.bbox ??
          Rect.fromLTWH(0, 0, _maskW.toDouble(), _maskH.toDouble());
      final newBbox = maskBbox ?? fallback;
      // Editor doesn't have the original capture in hand, so render the
      // post-edit preview from the working image. Detection-time previews
      // are still full-res; this only affects tiles that were touched
      // through the editor.
      final previewSource = pipe.decodeWorkingImage(widget.workingPng);
      final previewPng = pipe.renderInstancePreview(
        source: previewSource,
        mask: _mask,
        maskWidth: _maskW,
        maskHeight: _maskH,
        bbox: newBbox,
      );
      final repo = SampleRepository.instance;
      final SampleInstance saved;
      if (existing != null) {
        saved = await repo.updateInstance(existing.copyWith(
          bbox: newBbox,
          maskPng: maskPng,
          previewPng: previewPng,
          updatedAt: DateTime.now(),
        ));
      } else {
        final centroid = Offset(
          newBbox.left + newBbox.width / 2,
          newBbox.top + newBbox.height / 2,
        );
        final draft = SampleInstance(
          sampleId: widget.sampleId,
          idx: 0, // ignored — repo assigns the next idx atomically
          bbox: newBbox,
          centroid: centroid,
          score: 1.0,
          imageWidth: _maskW,
          imageHeight: _maskH,
          maskPng: maskPng,
          previewPng: previewPng,
          createdAt: DateTime.now(),
          updatedAt: DateTime.now(),
        );
        saved = await repo.createInstance(draft);
      }
      if (!mounted) return;
      Navigator.of(context).pop(InstanceEditorResult.saved(saved));
    } catch (e, st) {
      debugPrint('[editor] save failed: $e\n$st');
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _busy = false;
      });
    }
  }

  Future<void> _confirmDelete() async {
    final existing = widget.instance;
    if (existing == null || existing.id == null) return;
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: const Text('Delete instance?',
            style: TextStyle(color: Colors.white)),
        content: const Text(
          'This will permanently remove the segmentation mask for this '
          'instance.',
          style: TextStyle(color: Colors.white70),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Delete',
                style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
    if (ok != true) return;
    setState(() => _busy = true);
    try {
      await SampleRepository.instance.deleteInstance(existing.id!);
      if (!mounted) return;
      Navigator.of(context).pop(InstanceEditorResult.deleted(existing));
    } catch (e, st) {
      debugPrint('[editor] delete failed: $e\n$st');
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _busy = false;
      });
    }
  }

  /// Tightest axis-aligned rect around the binary mask, or null if empty.
  Rect? _bboxFromMask() {
    int minX = _maskW, minY = _maskH, maxX = -1, maxY = -1;
    for (int y = 0; y < _maskH; y++) {
      for (int x = 0; x < _maskW; x++) {
        if (_mask[y * _maskW + x] != 0) {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null;
    return Rect.fromLTRB(
      minX.toDouble(),
      minY.toDouble(),
      (maxX + 1).toDouble(),
      (maxY + 1).toDouble(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Text(
          widget.isCreating ? 'New instance' : 'Edit instance',
          style: const TextStyle(color: Colors.white, fontSize: 15),
        ),
        actions: [
          if (!widget.isCreating)
            IconButton(
              tooltip: 'Delete',
              onPressed: _busy ? null : _confirmDelete,
              icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
            ),
          TextButton(
            onPressed: _busy ? null : _save,
            child: Text(
              widget.isCreating
                  ? 'Create'
                  : (_dirty ? 'Save' : 'Done'),
              style: TextStyle(
                color: _busy ? Colors.white38 : Colors.white,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                Positioned.fill(child: _buildCanvas()),
                if (_busy)
                  Positioned.fill(
                    child: Container(
                      color: Colors.black54,
                      alignment: Alignment.center,
                      child: const CircularProgressIndicator(
                        color: Colors.white,
                      ),
                    ),
                  ),
                if (_error != null)
                  Positioned(
                    left: 16,
                    right: 16,
                    bottom: 16,
                    child: Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.redAccent.withValues(alpha: 0.85),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        _error!,
                        style: const TextStyle(
                            color: Colors.white, fontSize: 13),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          _buildToolbar(),
        ],
      ),
    );
  }

  Widget _buildCanvas() {
    final image = _fullImage;
    if (image == null) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }
    return LayoutBuilder(
      builder: (context, constraints) {
        // First-frame: center the viewer on the existing bbox so the user
        // lands looking at the instance instead of the working image's
        // top-left corner. We do this lazily because we need the viewport
        // size, which only LayoutBuilder knows.
        if (!_viewerInitialised) {
          _viewerInitialised = true;
          WidgetsBinding.instance.addPostFrameCallback((_) {
            _centerOnBbox(
              viewport: Size(constraints.maxWidth, constraints.maxHeight),
            );
          });
        }
        return InteractiveViewer(
          transformationController: _viewer,
          constrained: false,
          minScale: 0.1,
          maxScale: 8,
          // Allow panning anywhere in the working image with a generous margin
          // so the bbox can sit fully inside the viewport when zoomed in.
          boundaryMargin: const EdgeInsets.all(2000),
          panEnabled: _tool != _EditorTool.brush && _tool != _EditorTool.erase,
          child: SizedBox(
            width: _maskW.toDouble(),
            height: _maskH.toDouble(),
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTapDown: (d) => _onTap(d.localPosition),
              onPanStart: (d) => _onPan(d.localPosition, true),
              onPanUpdate: (d) => _onPan(d.localPosition, false),
              child: CustomPaint(
                painter: _CanvasPainter(
                  image: image,
                  mask: _mask,
                  maskW: _maskW,
                  maskH: _maskH,
                ),
                size: Size(_maskW.toDouble(), _maskH.toDouble()),
              ),
            ),
          ),
        );
      },
    );
  }

  void _centerOnBbox({required Size viewport}) {
    final bbox = widget.instance?.bbox ??
        Rect.fromLTWH(0, 0, _maskW.toDouble(), _maskH.toDouble());
    // Choose a scale that fits the bbox + a little context into the viewport.
    const padding = 64.0;
    final targetW = bbox.width + padding * 2;
    final targetH = bbox.height + padding * 2;
    final scale = (viewport.width / targetW)
        .clamp(0.1, 8.0)
        .toDouble();
    final scaleY = (viewport.height / targetH)
        .clamp(0.1, 8.0)
        .toDouble();
    final s = scale < scaleY ? scale : scaleY;
    // Translate so the bbox center maps to the viewport center.
    final cx = bbox.center.dx;
    final cy = bbox.center.dy;
    final tx = viewport.width / 2 - cx * s;
    final ty = viewport.height / 2 - cy * s;
    _viewer.value = Matrix4.identity()
      ..translateByDouble(tx, ty, 0, 1)
      ..scaleByDouble(s, s, 1, 1);
  }

  void _onTap(Offset local) {
    if (_tool == _EditorTool.sam) {
      _runSamPoint(local);
      return;
    }
    setState(() {
      _paintAt(local, inside: _tool == _EditorTool.brush);
    });
  }

  void _onPan(Offset local, bool isStart) {
    if (_tool == _EditorTool.sam) {
      // Single-shot prompt; ignore drags so we don't fire ten predicts.
      if (isStart) {
        _runSamPoint(local);
      }
      return;
    }
    setState(() {
      _paintAt(local, inside: _tool == _EditorTool.brush);
    });
  }

  Widget _buildToolbar() {
    return Container(
      color: const Color(0xFF111111),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                _ToolButton(
                  label: 'SAM',
                  icon: Icons.auto_awesome,
                  selected: _tool == _EditorTool.sam,
                  onTap: () {
                    setState(() => _tool = _EditorTool.sam);
                    _ensureEmbedding();
                  },
                ),
                const SizedBox(width: 8),
                _ToolButton(
                  label: 'Brush',
                  icon: Icons.brush,
                  selected: _tool == _EditorTool.brush,
                  onTap: () => setState(() => _tool = _EditorTool.brush),
                ),
                const SizedBox(width: 8),
                _ToolButton(
                  label: 'Erase',
                  icon: Icons.cleaning_services_outlined,
                  selected: _tool == _EditorTool.erase,
                  onTap: () => setState(() => _tool = _EditorTool.erase),
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (_tool == _EditorTool.sam)
              Row(
                children: [
                  const Text('Point:',
                      style: TextStyle(color: Colors.white70, fontSize: 12)),
                  const SizedBox(width: 8),
                  ChoiceChip(
                    label: const Text('Foreground'),
                    selected: _samLabel == 1,
                    onSelected: (_) => setState(() => _samLabel = 1),
                  ),
                  const SizedBox(width: 6),
                  ChoiceChip(
                    label: const Text('Background'),
                    selected: _samLabel == 0,
                    onSelected: (_) => setState(() => _samLabel = 0),
                  ),
                  if (_embeddingLoading) ...[
                    const Spacer(),
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white70,
                      ),
                    ),
                  ] else if (_embeddingError != null) ...[
                    const Spacer(),
                    const Icon(Icons.error_outline,
                        size: 16, color: Colors.redAccent),
                  ],
                ],
              )
            else
              Row(
                children: [
                  const Icon(Icons.lens, size: 14, color: Colors.white70),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Slider(
                      value: _brushRadius,
                      min: 2,
                      max: 64,
                      divisions: 31,
                      label: '${_brushRadius.round()} px',
                      onChanged: (v) => setState(() => _brushRadius = v),
                    ),
                  ),
                  Text('${_brushRadius.round()}',
                      style: const TextStyle(
                          color: Colors.white70, fontSize: 12)),
                ],
              ),
          ],
        ),
      ),
    );
  }
}

class _ToolButton extends StatelessWidget {
  const _ToolButton({
    required this.label,
    required this.icon,
    required this.selected,
    required this.onTap,
  });

  final String label;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: selected ? Colors.white : Colors.white12,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon,
                  size: 18, color: selected ? Colors.black : Colors.white),
              const SizedBox(height: 4),
              Text(
                label,
                style: TextStyle(
                  color: selected ? Colors.black : Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _CanvasPainter extends CustomPainter {
  _CanvasPainter({
    required this.image,
    required this.mask,
    required this.maskW,
    required this.maskH,
  });

  final ui.Image image;
  final Uint8List mask;
  final int maskW;
  final int maskH;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..filterQuality = FilterQuality.medium;
    final src =
        Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final dst = Offset.zero & size;
    canvas.drawImageRect(image, src, dst, paint);

    // Tinted mask overlay + outline. Drawn pixel-by-pixel through a
    // PictureRecorder would be too slow; we instead trace the mask once
    // into a Path (run-by-run rectangles) so Skia handles the rasterisation.
    final fill = Paint()
      ..color = const Color.fromARGB(110, 0, 230, 118)
      ..style = PaintingStyle.fill;
    final stroke = Paint()
      ..color = const Color.fromARGB(255, 0, 230, 118)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    // Canvas is sized at the mask's pixel dimensions (1:1), so no scaling.
    final path = Path();
    final outline = Path();
    bool inside(int mx, int my) {
      if (mx < 0 || my < 0 || mx >= maskW || my >= maskH) return false;
      return mask[my * maskW + mx] != 0;
    }

    for (int y = 0; y < maskH; y++) {
      int run = -1;
      for (int x = 0; x < maskW; x++) {
        final on = inside(x, y);
        if (on && run < 0) run = x;
        if ((!on || x == maskW - 1) && run >= 0) {
          final endX = on ? x + 1 : x;
          path.addRect(Rect.fromLTWH(
            run.toDouble(),
            y.toDouble(),
            (endX - run).toDouble(),
            1,
          ));
          run = -1;
        }
        if (on) {
          final isEdge = !inside(x - 1, y) ||
              !inside(x + 1, y) ||
              !inside(x, y - 1) ||
              !inside(x, y + 1);
          if (isEdge) {
            outline.addRect(
                Rect.fromLTWH(x.toDouble(), y.toDouble(), 1, 1));
          }
        }
      }
    }
    canvas.drawPath(path, fill);
    canvas.drawPath(outline, stroke);
  }

  @override
  bool shouldRepaint(covariant _CanvasPainter old) =>
      old.image != image || !identical(old.mask, mask);
}
