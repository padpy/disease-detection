import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:gopher_eye/model/detection_mode.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/model/sample_instance.dart';
import 'package:gopher_eye/screens/instance_editor_screen.dart';
import 'package:gopher_eye/services/grape_leaf_pipeline.dart' show grapeLeafDisplayLabel;
import 'package:gopher_eye/services/sample_repository.dart';
import 'package:gopher_eye/widgets/sample_tag_edit_dialog.dart';
import 'package:url_launcher/url_launcher.dart';

/// What [InstanceInspectorScreen] returns to its caller. The inspector can
/// surface two independent kinds of edits in one visit (the instance editor
/// and the sample-tag editor); fields are populated only for the edits the
/// user actually performed so the caller can avoid re-running disease
/// analysis when only sample metadata changed.
class InspectorResult {
  const InspectorResult({this.editorResult, this.sample});

  /// Result forwarded from the instance editor (mask edit or instance
  /// deletion). Null when the user never opened the editor or cancelled.
  final InstanceEditorResult? editorResult;

  /// Updated sample row, set when the user edited the QR-derived "Sample
  /// tag" fields from inside the inspector.
  final Sample? sample;

  bool get isEmpty => editorResult == null && sample == null;
}

/// Read-only details view for a single [SampleInstance]. Shows the rendered
/// preview tile (and disease overlay tile if available) along with bbox,
/// score, and FHB stats. Tap "Edit" in the app bar to push the editor; the
/// editor's result is forwarded back to the caller.
class InstanceInspectorScreen extends StatefulWidget {
  const InstanceInspectorScreen({
    super.key,
    required this.sample,
    required this.instance,
    required this.indexLabel,
    required this.workingPng,
    required this.workingWidth,
    required this.workingHeight,
  });

  final Sample sample;
  final SampleInstance instance;

  /// 1-based label as shown in the strip (the strip uses the position in the
  /// list, not [SampleInstance.idx], so we let the caller compute it).
  final int indexLabel;

  final Uint8List workingPng;
  final int workingWidth;
  final int workingHeight;

  @override
  State<InstanceInspectorScreen> createState() =>
      _InstanceInspectorScreenState();
}

/// Three preview states for the inspector's instance tile: the raw cropped
/// photo, the segmentation outline, or the disease classification overlay.
/// "Off" lets the user inspect the raw pixels without any annotation.
enum _PreviewMode { off, segmentation, disease }

class _InstanceInspectorScreenState extends State<InstanceInspectorScreen> {
  late SampleInstance _instance = widget.instance;
  late Sample _sample = widget.sample;
  Sample? _sampleEdit; // non-null once the user has saved a QR edit
  late _PreviewMode _previewMode = widget.instance.diseasePreviewPng != null
      ? _PreviewMode.disease
      : _PreviewMode.segmentation;
  bool _busy = false;
  String? _error;

  Future<void> _openMaps() async {
    if (!_sample.hasLocation) return;
    final lat = _sample.latitude!;
    final lng = _sample.longitude!;
    final uri = Platform.isIOS
        ? Uri.parse('https://maps.apple.com/?ll=$lat,$lng&q=Sample')
        : Uri.parse('geo:$lat,$lng?q=$lat,$lng(Sample)');
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  void _popWithResult(InstanceEditorResult? editorResult) {
    final result = InspectorResult(
      editorResult: editorResult,
      sample: _sampleEdit,
    );
    Navigator.of(context).pop(result.isEmpty ? null : result);
  }

  Future<void> _edit() async {
    final result = await Navigator.of(context).push<InstanceEditorResult>(
      MaterialPageRoute(
        builder: (_) => InstanceEditorScreen(
          instance: _instance,
          workingPng: widget.workingPng,
          workingWidth: widget.workingWidth,
          workingHeight: widget.workingHeight,
        ),
      ),
    );
    if (result == null) return;
    if (!mounted) return;
    if (result.deleted) {
      _popWithResult(result);
      return;
    }
    setState(() => _instance = result.instance);
    _popWithResult(result);
  }

  Future<void> _confirmDelete() async {
    if (_instance.id == null) return;
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
      await SampleRepository.instance.deleteInstance(_instance.id!);
      if (!mounted) return;
      _popWithResult(InstanceEditorResult.deleted(_instance));
    } catch (e, st) {
      debugPrint('[inspector] delete failed: $e\n$st');
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _busy = false;
      });
    }
  }

  Future<void> _editSampleTag() async {
    if (_sample.id == null) return;
    final edited = await showDialog<SampleTagDraft>(
      context: context,
      builder: (_) => SampleTagEditDialog(initial: _sample),
    );
    if (edited == null || !mounted) return;
    setState(() => _busy = true);
    try {
      final updated = await SampleRepository.instance.updateQrMetadata(
        sampleId: _sample.id!,
        qrId: edited.qrId,
        qrLine: edited.qrLine,
        qrRep: edited.qrRep,
        qrLocation: edited.qrLocation,
        qrNote: edited.qrNote,
      );
      if (!mounted || updated == null) return;
      setState(() {
        _sample = updated;
        _sampleEdit = updated;
        _busy = false;
      });
    } catch (e, st) {
      debugPrint('[inspector] sample tag save failed: $e\n$st');
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _busy = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope<Object?>(
      // Always allow the pop, but if the user backed out without going
      // through _edit/_confirmDelete we still need to return any
      // sample-tag edit that happened in this session.
      canPop: _sampleEdit == null,
      onPopInvokedWithResult: (didPop, _) {
        if (didPop) return;
        _popWithResult(null);
      },
      child: Scaffold(
        backgroundColor: Colors.black,
        appBar: AppBar(
          backgroundColor: Colors.black,
          iconTheme: const IconThemeData(color: Colors.white),
          title: Text(
            'Instance #${widget.indexLabel}',
            style: const TextStyle(color: Colors.white, fontSize: 15),
          ),
          actions: [
            IconButton(
              tooltip: 'Delete',
              onPressed: _busy ? null : _confirmDelete,
              icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
            ),
            TextButton.icon(
              onPressed: _busy ? null : _edit,
              icon: Icon(Icons.edit,
                  size: 18,
                  color: _busy ? Colors.white38 : Colors.white),
              label: Text(
                'Edit',
                style: TextStyle(
                  color: _busy ? Colors.white38 : Colors.white,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
        body: Stack(
          children: [
            SafeArea(
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  _buildPreview(),
                  const SizedBox(height: 12),
                  _buildPreviewToggle(),
                  const SizedBox(height: 20),
                  _buildDetails(),
                ],
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
                    style: const TextStyle(color: Colors.white, fontSize: 13),
                  ),
                ),
              ),
            if (_busy)
              const Positioned.fill(
                child: ColoredBox(
                  color: Colors.black54,
                  child: Center(
                    child: CircularProgressIndicator(color: Colors.white),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildPreview() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: AspectRatio(
        aspectRatio: _instance.bbox.width / _instance.bbox.height,
        child: InteractiveViewer(
          minScale: 1,
          maxScale: 5,
          child: switch (_previewMode) {
            _PreviewMode.off => _RawBboxCrop(
                workingPng: widget.workingPng,
                workingWidth: widget.workingWidth,
                workingHeight: widget.workingHeight,
                bbox: _instance.bbox,
              ),
            _PreviewMode.segmentation => Image.memory(
                _instance.previewPng,
                gaplessPlayback: true,
                fit: BoxFit.contain,
              ),
            _PreviewMode.disease => Image.memory(
                _instance.diseasePreviewPng ?? _instance.previewPng,
                gaplessPlayback: true,
                fit: BoxFit.contain,
              ),
          },
        ),
      ),
    );
  }

  Widget _buildPreviewToggle() {
    final hasDisease = _instance.diseasePreviewPng != null;
    return SegmentedButton<_PreviewMode>(
      showSelectedIcon: false,
      segments: [
        const ButtonSegment(
          value: _PreviewMode.off,
          label: Text('Off'),
          icon: Icon(Icons.visibility_off_outlined),
        ),
        const ButtonSegment(
          value: _PreviewMode.segmentation,
          label: Text('Segment'),
          icon: Icon(Icons.gesture),
        ),
        ButtonSegment(
          value: _PreviewMode.disease,
          label: const Text('Disease'),
          icon: const Icon(Icons.local_florist),
          enabled: hasDisease,
        ),
      ],
      selected: {_previewMode},
      onSelectionChanged: (s) => setState(() => _previewMode = s.first),
      style: ButtonStyle(
        foregroundColor: WidgetStateProperty.resolveWith(
            (s) => s.contains(WidgetState.selected)
                ? Colors.black
                : Colors.white70),
        backgroundColor: WidgetStateProperty.resolveWith(
            (s) => s.contains(WidgetState.selected)
                ? Colors.white
                : Colors.white10),
        side: WidgetStateProperty.all(
            const BorderSide(color: Colors.white24)),
      ),
    );
  }

  Widget _buildDetails() {
    // Grape-leaf samples are reported as a disease classification only —
    // the YOLO confidence and the binary 0/1 FHB ratio aren't meaningful
    // for the SwinV2 3-class label and would be misleading.
    final isGrape = _sample.detectionMode == DetectionMode.grapeLeaf;
    final bbox = _instance.bbox;
    final rows = <_DetailRow>[
      if (!isGrape)
        _DetailRow('Score', '${(_instance.score * 100).toStringAsFixed(1)}%'),
      _DetailRow(
        'Bbox',
        '(${bbox.left.toStringAsFixed(0)}, ${bbox.top.toStringAsFixed(0)}) · '
            '${bbox.width.toStringAsFixed(0)} × ${bbox.height.toStringAsFixed(0)} px',
      ),
      _DetailRow(
        'Centroid',
        '(${_instance.centroid.dx.toStringAsFixed(0)}, '
            '${_instance.centroid.dy.toStringAsFixed(0)})',
      ),
      _DetailRow('Updated', _formatDateTime(_instance.updatedAt)),
    ];

    final fhbRows = <_DetailRow>[];
    if (!isGrape && _instance.fhbRatio != null) {
      fhbRows.add(_DetailRow(
        'FHB ratio',
        '${(_instance.fhbRatio! * 100).toStringAsFixed(1)}%',
      ));
    }
    if (_instance.fhbSeverity != null) {
      fhbRows.add(_DetailRow(
        isGrape ? 'Disease' : 'Severity',
        isGrape
            ? grapeLeafDisplayLabel(_instance.fhbSeverity!)
            : _instance.fhbSeverity!,
      ));
    }
    if (_instance.fhbGreenCount != null) {
      fhbRows.add(_DetailRow(
        'Healthy px',
        _instance.fhbGreenCount!.toString(),
      ));
    }
    if (_instance.fhbNecroticCount != null) {
      fhbRows.add(_DetailRow(
        'Necrotic px',
        _instance.fhbNecroticCount!.toString(),
      ));
    }
    if (_instance.fhbOtherCount != null) {
      fhbRows.add(_DetailRow(
        'Other px',
        _instance.fhbOtherCount!.toString(),
      ));
    }
    if (_instance.fhbTotalPixels != null) {
      fhbRows.add(_DetailRow(
        'Total mask px',
        _instance.fhbTotalPixels!.toString(),
      ));
    }

    final qrRows = <_DetailRow>[];
    final s = _sample;
    if (s.qrId != null && s.qrId!.isNotEmpty) {
      qrRows.add(_DetailRow('ID', s.qrId!));
    }
    if (s.qrLine != null && s.qrLine!.isNotEmpty) {
      qrRows.add(_DetailRow('Line', s.qrLine!));
    }
    if (s.qrRep != null && s.qrRep!.isNotEmpty) {
      qrRows.add(_DetailRow('Rep', s.qrRep!));
    }
    if (s.qrLocation != null && s.qrLocation!.isNotEmpty) {
      qrRows.add(_DetailRow('Location', s.qrLocation!));
    }
    if (s.qrNote != null && s.qrNote!.isNotEmpty) {
      qrRows.add(_DetailRow('Note', s.qrNote!));
    }

    final canEditTag = _sample.id != null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const _SectionHeader(title: 'Detection'),
        ...rows,
        if (fhbRows.isNotEmpty) ...[
          const SizedBox(height: 16),
          const _SectionHeader(title: 'Disease analysis'),
          ...fhbRows,
        ],
        if (_sample.hasLocation) ...[
          const SizedBox(height: 16),
          const _SectionHeader(title: 'GPS'),
          _GpsRow(
            latitude: _sample.latitude!,
            longitude: _sample.longitude!,
            onTap: _busy ? null : _openMaps,
          ),
        ],
        if (canEditTag) ...[
          const SizedBox(height: 16),
          _SectionHeader(
            title: 'Sample tag',
            trailing: TextButton.icon(
              onPressed: _busy ? null : _editSampleTag,
              icon: const Icon(Icons.edit, size: 16, color: Colors.white),
              label: const Text(
                'Edit',
                style: TextStyle(color: Colors.white, fontSize: 13),
              ),
              style: TextButton.styleFrom(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 0),
                minimumSize: const Size(0, 28),
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
            ),
          ),
          if (qrRows.isEmpty)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 4),
              child: Text(
                'No tag attached. Tap Edit to add an ID, line, rep, '
                'location, or note.',
                style: TextStyle(color: Colors.white54, fontSize: 13),
              ),
            )
          else
            ...qrRows,
        ],
      ],
    );
  }
}

/// Shows the sample's captured GPS coordinates with a "Open in Maps" link.
/// The samples list intentionally renders coords as static text, so this
/// inspector is the only place in the app where tapping launches the
/// platform maps app.
class _GpsRow extends StatelessWidget {
  const _GpsRow({
    required this.latitude,
    required this.longitude,
    required this.onTap,
  });

  final double latitude;
  final double longitude;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: GestureDetector(
        onTap: onTap,
        behavior: HitTestBehavior.opaque,
        child: Row(
          children: [
            const SizedBox(
              width: 110,
              child: Text(
                'Location',
                style: TextStyle(color: Colors.white54, fontSize: 13),
              ),
            ),
            const Icon(
              Icons.place_outlined,
              size: 14,
              color: Colors.lightBlueAccent,
            ),
            const SizedBox(width: 4),
            Expanded(
              child: Text(
                '${latitude.toStringAsFixed(5)}, ${longitude.toStringAsFixed(5)}',
                style: const TextStyle(
                  color: Colors.lightBlueAccent,
                  fontSize: 13,
                  decoration: TextDecoration.underline,
                ),
              ),
            ),
            const Icon(
              Icons.open_in_new,
              size: 14,
              color: Colors.lightBlueAccent,
            ),
          ],
        ),
      ),
    );
  }
}

/// Renders only the bbox region of the working image, with no overlay drawn on
/// top. Used when the inspector's preview-mode toggle is set to "Off" so the
/// user can inspect the raw pixels without segmentation or disease coloring
/// occluding them. The working image is positioned/scaled so the requested
/// bbox fills the available width.
class _RawBboxCrop extends StatelessWidget {
  const _RawBboxCrop({
    required this.workingPng,
    required this.workingWidth,
    required this.workingHeight,
    required this.bbox,
  });

  final Uint8List workingPng;
  final int workingWidth;
  final int workingHeight;
  final Rect bbox;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final scale = constraints.maxWidth / bbox.width;
        return ClipRect(
          child: OverflowBox(
            minWidth: 0,
            maxWidth: double.infinity,
            minHeight: 0,
            maxHeight: double.infinity,
            alignment: Alignment.topLeft,
            child: Transform.translate(
              offset: Offset(-bbox.left * scale, -bbox.top * scale),
              child: SizedBox(
                width: workingWidth * scale,
                height: workingHeight * scale,
                child: Image.memory(
                  workingPng,
                  fit: BoxFit.fill,
                  gaplessPlayback: true,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.title, this.trailing});
  final String title;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final label = Text(
      title,
      style: const TextStyle(
        color: Colors.white,
        fontSize: 13,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.3,
      ),
    );
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: trailing == null
          ? label
          : Row(
              children: [
                Expanded(child: label),
                trailing!,
              ],
            ),
    );
  }
}

class _DetailRow extends StatelessWidget {
  const _DetailRow(this.label, this.value);
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 110,
            child: Text(
              label,
              style: const TextStyle(color: Colors.white54, fontSize: 13),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(color: Colors.white, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}

String _formatDateTime(DateTime dt) {
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  final hour12 = dt.hour == 0
      ? 12
      : (dt.hour > 12 ? dt.hour - 12 : dt.hour);
  final minute = dt.minute.toString().padLeft(2, '0');
  final ampm = dt.hour >= 12 ? 'PM' : 'AM';
  return '${months[dt.month - 1]} ${dt.day}, ${dt.year} · $hour12:$minute $ampm';
}

