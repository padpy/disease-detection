import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:gopher_eye/model/collection.dart';
import 'package:gopher_eye/model/detection_mode.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/model/sample_instance.dart';
import 'package:gopher_eye/screens/collection_picker_screen.dart';
import 'package:gopher_eye/screens/collection_samples_screen.dart';
import 'package:gopher_eye/screens/export_screen.dart';
import 'package:gopher_eye/screens/instance_editor_screen.dart';
import 'package:gopher_eye/screens/instance_inspector_screen.dart';
import 'package:gopher_eye/services/detection_service.dart';
import 'package:gopher_eye/services/grape_leaf_pipeline.dart';
import 'package:gopher_eye/services/sample_repository.dart';
import 'package:gopher_eye/widgets/sample_tag_edit_dialog.dart';
import 'package:url_launcher/url_launcher.dart';

class SamplesScreen extends StatefulWidget {
  const SamplesScreen({super.key, this.onBack});

  final VoidCallback? onBack;

  @override
  State<SamplesScreen> createState() => _SamplesScreenState();
}

/// One entry in the grouped samples list. Either a collection (which opens
/// its own drill-in screen on tap) or a single uncollected sample.
sealed class _SamplesListEntry {
  const _SamplesListEntry();
  DateTime get sortKey;
}

class _CollectionEntry extends _SamplesListEntry {
  const _CollectionEntry({
    required this.collection,
    required this.sampleCount,
    required this.lastSampleAt,
    this.coverImagePath,
  });

  final Collection collection;
  final int sampleCount;

  /// Most recent capture in the collection, or the collection's createdAt
  /// when empty. Used for the entry's secondary line and as the sort key so
  /// active collections float to the top.
  final DateTime lastSampleAt;
  final String? coverImagePath;

  @override
  DateTime get sortKey => lastSampleAt;
}

class _SampleEntry extends _SamplesListEntry {
  const _SampleEntry(this.sample);
  final Sample sample;

  @override
  DateTime get sortKey => sample.takenAt;
}

class _SamplesScreenState extends State<SamplesScreen> {
  late Future<List<_SamplesListEntry>> _future;
  List<_SamplesListEntry> _entries = [];

  @override
  void initState() {
    super.initState();
    _future = _load();
    DetectionService.instance.addListener(_onAnyJobCompleted);
  }

  @override
  void dispose() {
    DetectionService.instance.removeListener(_onAnyJobCompleted);
    super.dispose();
  }

  void _onAnyJobCompleted() {
    // The service notifies listeners when a job completes/fails so the list
    // can pick up newly-saved instances. The per-tile status row already
    // updates itself via its own ValueListenable; this hook is just a nudge
    // for any list-level state that might depend on completion.
    if (mounted) setState(() {});
  }

  Future<List<_SamplesListEntry>> _load() async {
    try {
      final repo = SampleRepository.instance;
      final samples = await repo.listAll();
      final collections = await repo.listCollections();
      debugPrint(
        '[samples] loaded ${samples.length} samples, '
        '${collections.length} collections',
      );

      // Bucket samples by collection_id so we can build the cover thumb +
      // count + lastSampleAt without re-querying per collection.
      final byCollection = <int, List<Sample>>{};
      final uncollected = <Sample>[];
      for (final s in samples) {
        if (s.collectionId == null) {
          uncollected.add(s);
        } else {
          (byCollection[s.collectionId!] ??= []).add(s);
        }
      }

      final entries = <_SamplesListEntry>[
        for (final c in collections)
          _CollectionEntry(
            collection: c,
            sampleCount: byCollection[c.id]?.length ?? 0,
            lastSampleAt:
                (byCollection[c.id]?.isNotEmpty ?? false)
                    ? byCollection[c.id]!.first.takenAt
                    : c.createdAt,
            coverImagePath:
                (byCollection[c.id]?.isNotEmpty ?? false)
                    ? byCollection[c.id]!.first.filePath
                    : null,
          ),
        for (final s in uncollected) _SampleEntry(s),
      ]..sort((a, b) => b.sortKey.compareTo(a.sortKey));

      if (mounted) setState(() => _entries = entries);
      return entries;
    } catch (e, st) {
      debugPrint('[samples] load failed: $e\n$st');
      rethrow;
    }
  }

  void _refresh() {
    setState(() {
      _entries = [];
      _future = _load();
    });
  }

  Future<void> _delete(Sample sample) async {
    if (sample.id == null) return;
    await SampleRepository.instance.delete(sample.id!);
    DetectionService.instance.forget(sample.id!);
    if (!mounted) return;
    setState(() => _entries.removeWhere(
        (e) => e is _SampleEntry && e.sample.id == sample.id));
  }

  Future<void> _openCollection(_CollectionEntry entry) async {
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) =>
            CollectionSamplesScreen(collection: entry.collection),
      ),
    );
    // Refresh on return — the user may have reassigned or deleted samples
    // inside the collection.
    if (mounted) _refresh();
  }

  Future<bool> _confirmDelete(Sample sample) async {
    final result = await showModalBottomSheet<bool>(
      context: context,
      backgroundColor: Colors.grey[900],
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Padding(
              padding: EdgeInsets.all(16),
              child: Text(
                'Delete this sample?',
                style: TextStyle(color: Colors.white, fontSize: 16),
              ),
            ),
            ListTile(
              leading: const Icon(Icons.delete_outline, color: Colors.redAccent),
              title: const Text(
                'Delete',
                style: TextStyle(color: Colors.redAccent),
              ),
              onTap: () => Navigator.of(ctx).pop(true),
            ),
            ListTile(
              leading: const Icon(Icons.close, color: Colors.white70),
              title: const Text(
                'Cancel',
                style: TextStyle(color: Colors.white70),
              ),
              onTap: () => Navigator.of(ctx).pop(false),
            ),
          ],
        ),
      ),
    );
    return result ?? false;
  }

  Future<void> _openMaps(Sample sample) async {
    if (!sample.hasLocation) return;
    final lat = sample.latitude!;
    final lng = sample.longitude!;
    final uri = Platform.isIOS
        ? Uri.parse('https://maps.apple.com/?ll=$lat,$lng&q=Sample')
        : Uri.parse('geo:$lat,$lng?q=$lat,$lng(Sample)');
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  void _openViewer(Sample sample) {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => SampleViewerScreen(sample: sample)),
    );
  }

  Future<void> _openExport() async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const ExportScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text('Samples', style: TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
        leading: widget.onBack != null
            ? IconButton(
                icon: const Icon(Icons.camera_alt_outlined, color: Colors.white),
                tooltip: 'Camera',
                onPressed: widget.onBack,
              )
            : null,
        actions: [
          IconButton(
            icon: const Icon(Icons.ios_share, color: Colors.white),
            tooltip: 'Export',
            onPressed: _openExport,
          ),
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: _refresh,
          ),
        ],
      ),
      body: FutureBuilder<List<_SamplesListEntry>>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(
              child: CircularProgressIndicator(color: Colors.white),
            );
          }
          if (snapshot.hasError) {
            return Padding(
              padding: const EdgeInsets.all(24),
              child: Center(
                child: Text(
                  'Failed to load samples:\n${snapshot.error}',
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.redAccent),
                ),
              ),
            );
          }
          if (_entries.isEmpty) {
            return const Center(
              child: Text(
                'No samples yet — capture some plants',
                style: TextStyle(color: Colors.white70),
              ),
            );
          }
          return ListView.separated(
            itemCount: _entries.length,
            separatorBuilder: (_, __) =>
                const Divider(height: 1, color: Colors.white12),
            itemBuilder: (context, index) {
              final entry = _entries[index];
              if (entry is _CollectionEntry) {
                return CollectionTile(
                  collection: entry.collection,
                  sampleCount: entry.sampleCount,
                  lastSampleAt: entry.lastSampleAt,
                  coverImagePath: entry.coverImagePath,
                  onTap: () => _openCollection(entry),
                );
              }
              final sample = (entry as _SampleEntry).sample;
              return Dismissible(
                key: ValueKey('sample-${sample.id}'),
                direction: DismissDirection.endToStart,
                background: Container(
                  color: Colors.redAccent,
                  alignment: Alignment.centerRight,
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: const Icon(Icons.delete, color: Colors.white),
                ),
                confirmDismiss: (_) => _confirmDelete(sample),
                onDismissed: (_) => _delete(sample),
                child: SampleTile(
                  sample: sample,
                  onTap: () => _openViewer(sample),
                  onLongPress: () async {
                    if (await _confirmDelete(sample)) await _delete(sample);
                  },
                  onCoordsTap: () => _openMaps(sample),
                ),
              );
            },
          );
        },
      ),
    );
  }
}

/// Single-row entry for a collection in the samples list. Shows the cover
/// thumbnail (latest sample image, or placeholder when empty), name, sample
/// count, and the timestamp of the latest capture.
class CollectionTile extends StatelessWidget {
  const CollectionTile({
    super.key,
    required this.collection,
    required this.sampleCount,
    required this.lastSampleAt,
    required this.coverImagePath,
    required this.onTap,
  });

  final Collection collection;
  final int sampleCount;
  final DateTime lastSampleAt;
  final String? coverImagePath;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: SizedBox(
                width: 64,
                height: 64,
                child: coverImagePath != null
                    ? Image.file(
                        File(coverImagePath!),
                        fit: BoxFit.cover,
                        cacheWidth: 192,
                        cacheHeight: 192,
                        errorBuilder: (_, __, ___) =>
                            _CollectionPlaceholder(),
                      )
                    : _CollectionPlaceholder(),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(
                        Icons.collections_bookmark_outlined,
                        size: 14,
                        color: Colors.amberAccent,
                      ),
                      const SizedBox(width: 4),
                      Expanded(
                        child: Text(
                          collection.name,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: Colors.white12,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                              color: Colors.white24, width: 0.8),
                        ),
                        child: Text(
                          '$sampleCount',
                          style: const TextStyle(
                              color: Colors.white70, fontSize: 11),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    sampleCount == 0
                        ? 'Empty · created ${_formatDate(collection.createdAt)}'
                        : 'Latest ${_formatDate(lastSampleAt)}',
                    style: const TextStyle(
                        color: Colors.white54, fontSize: 13),
                  ),
                ],
              ),
            ),
            const Icon(Icons.chevron_right, color: Colors.white38),
          ],
        ),
      ),
    );
  }
}

class _CollectionPlaceholder extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white10,
      alignment: Alignment.center,
      child: const Icon(
        Icons.collections_bookmark_outlined,
        color: Colors.white38,
      ),
    );
  }
}

class SampleTile extends StatelessWidget {
  const SampleTile({
    super.key,
    required this.sample,
    required this.onTap,
    required this.onLongPress,
    required this.onCoordsTap,
  });

  final Sample sample;
  final VoidCallback onTap;
  final VoidCallback onLongPress;
  final VoidCallback onCoordsTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      onLongPress: onLongPress,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.file(
                File(sample.filePath),
                width: 64,
                height: 64,
                fit: BoxFit.cover,
                cacheWidth: 192,
                cacheHeight: 192,
                errorBuilder: (_, __, ___) => Container(
                  width: 64,
                  height: 64,
                  color: Colors.grey[800],
                  child: const Icon(
                    Icons.broken_image_outlined,
                    color: Colors.white54,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          _formatDate(sample.takenAt),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 15,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                      _ModeBadge(mode: sample.detectionMode),
                    ],
                  ),
                  const SizedBox(height: 4),
                  if (sample.hasLocation)
                    GestureDetector(
                      onTap: onCoordsTap,
                      behavior: HitTestBehavior.opaque,
                      child: Row(
                        children: [
                          const Icon(
                            Icons.place_outlined,
                            size: 14,
                            color: Colors.lightBlueAccent,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            '${sample.latitude!.toStringAsFixed(4)}, ${sample.longitude!.toStringAsFixed(4)}',
                            style: const TextStyle(
                              color: Colors.lightBlueAccent,
                              fontSize: 13,
                              decoration: TextDecoration.underline,
                            ),
                          ),
                        ],
                      ),
                    )
                  else
                    const Text(
                      'No location',
                      style: TextStyle(color: Colors.white38, fontSize: 13),
                    ),
                  if (sample.hasQrMetadata) ...[
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        const Icon(
                          Icons.qr_code,
                          size: 14,
                          color: Colors.white70,
                        ),
                        const SizedBox(width: 4),
                        Expanded(
                          child: Text(
                            _qrLabel(sample),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                  if (sample.id != null) ...[
                    const SizedBox(height: 6),
                    _SampleJobStatusRow(
                      sampleId: sample.id!,
                      mode: sample.detectionMode,
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Tiny pill showing a sample's detection mode in the list — purely visual,
/// not tappable. Mode switching happens in the sample viewer.
class _ModeBadge extends StatelessWidget {
  const _ModeBadge({required this.mode});

  final DetectionMode mode;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.white12,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white24, width: 0.8),
      ),
      child: Text(
        mode.label,
        style: const TextStyle(color: Colors.white70, fontSize: 11),
      ),
    );
  }
}

class SampleViewerScreen extends StatefulWidget {
  const SampleViewerScreen({super.key, required this.sample});

  final Sample sample;

  @override
  State<SampleViewerScreen> createState() => _SampleViewerScreenState();
}

enum _OverlayMode { bbox, segmentation, disease }

class _SampleViewerScreenState extends State<SampleViewerScreen> {
  String? _error;
  bool _showOverlay = true;
  _OverlayMode _overlayMode = _OverlayMode.segmentation;

  /// Local copy so we can reflect mode switches without leaving the screen.
  late Sample _sample = widget.sample;

  /// Resolved collection for the current sample (null when uncollected).
  /// Hydrated lazily on screen load + after the user picks a different one
  /// from the panel.
  Collection? _collection;

  /// Working-image PNG bytes — kept in state so the editor (which still
  /// operates at SAM resolution) can be opened without a re-decode. The
  /// viewer itself displays the original capture; this is just bookkeeping
  /// metadata to know we have detection data.
  Uint8List? _workingPng;
  int? _workingW;
  int? _workingH;

  /// Persisted, ordered by `idx`. Source of truth for the on-screen list.
  List<SampleInstance> _instances = [];
  Duration? _lastElapsed;

  /// Pre-rendered combined overlays (working-image resolution), stretched
  /// to fit over the original capture which has the same aspect ratio.
  Uint8List? _segmentationOverlayPng;
  Uint8List? _diseaseOverlayPng;

  ValueListenable<DetectionStatus>? _jobStatus;
  void Function()? _jobListener;

  @override
  void initState() {
    super.initState();
    _hydrate();
    _attachJobListener();
  }

  void _attachJobListener() {
    final id = _sample.id;
    if (id == null) return;
    _jobStatus = DetectionService.instance.statusFor(id);
    _jobListener = () {
      final status = _jobStatus!.value;
      if (status is DetectionCompleted) {
        // Pipeline just finished — pull the freshly-persisted state.
        _hydrate(elapsed: status.elapsed);
      } else if (status is DetectionFailed) {
        if (mounted) setState(() => _error = status.error);
      } else if (status is DetectionRunning) {
        if (mounted) setState(() => _error = null);
      }
    };
    _jobStatus!.addListener(_jobListener!);
  }

  @override
  void dispose() {
    if (_jobStatus != null && _jobListener != null) {
      _jobStatus!.removeListener(_jobListener!);
    }
    super.dispose();
  }

  Future<void> _hydrate({Duration? elapsed}) async {
    if (_sample.id == null) return;
    final repo = SampleRepository.instance;
    // Run all hydration queries in parallel — they target three different
    // tables/blob columns and don't depend on each other.
    final results = await Future.wait([
      repo.loadWorkingImage(_sample.id!),
      repo.listInstances(_sample.id!),
      repo.loadDiseaseOverlay(_sample.id!),
      repo.loadSegmentationOverlay(_sample.id!),
      if (_sample.collectionId != null)
        repo.findCollection(_sample.collectionId!),
    ]);
    final wi = results[0] as ({Uint8List png, int width, int height})?;
    final list = results[1] as List<SampleInstance>;
    final disease = results[2] as Uint8List?;
    final segmentation = results[3] as Uint8List?;
    final collection = _sample.collectionId == null
        ? null
        : results[4] as Collection?;
    if (!mounted) return;
    setState(() {
      if (wi != null) {
        _workingPng = wi.png;
        _workingW = wi.width;
        _workingH = wi.height;
      }
      _instances = list;
      _diseaseOverlayPng = disease;
      _segmentationOverlayPng = segmentation;
      _collection = collection;
      if (elapsed != null) _lastElapsed = elapsed;
    });
  }

  Future<void> _editSampleTag() async {
    if (_sample.id == null) return;
    final edited = await showDialog<SampleTagDraft>(
      context: context,
      builder: (_) => SampleTagEditDialog(initial: _sample),
    );
    if (edited == null || !mounted) return;
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
      setState(() => _sample = updated);
    } catch (e, st) {
      debugPrint('[viewer] sample tag save failed: $e\n$st');
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _changeCollection() async {
    if (_sample.id == null) return;
    final result = await Navigator.of(context).push<CollectionPickResult>(
      MaterialPageRoute(
        builder: (_) => CollectionPickerScreen(
          activeCollectionId: _collection?.id,
        ),
      ),
    );
    if (result == null || !mounted) return;
    final updated = await SampleRepository.instance.setSampleCollection(
      sampleId: _sample.id!,
      collectionId: result.collection?.id,
    );
    if (!mounted || updated == null) return;
    setState(() {
      _sample = updated;
      _collection = result.collection;
    });
  }

  void _detect() {
    if (_sample.id == null) return;
    setState(() => _error = null);
    DetectionService.instance.requeue(
      sampleId: _sample.id!,
      filePath: _sample.filePath,
      mode: _sample.detectionMode,
    );
  }

  Future<void> _pickMode() async {
    final picked = await showModalBottomSheet<DetectionMode>(
      context: context,
      backgroundColor: Colors.grey[900],
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Detection mode',
                  style: TextStyle(color: Colors.white, fontSize: 14),
                ),
              ),
            ),
            for (final m in DetectionMode.values)
              ListTile(
                leading: Icon(
                  m == _sample.detectionMode
                      ? Icons.radio_button_checked
                      : Icons.radio_button_unchecked,
                  color: Colors.white,
                ),
                title: Text(
                  m.longLabel,
                  style: const TextStyle(color: Colors.white),
                ),
                subtitle: Text(
                  m.hasAutoDetection
                      ? 'Auto-detects ${m.instanceNounPlural}'
                      : 'Add ${m.instanceNounPlural} manually with the + tile',
                  style: const TextStyle(color: Colors.white54, fontSize: 12),
                ),
                onTap: () => Navigator.of(ctx).pop(m),
              ),
          ],
        ),
      ),
    );
    if (picked != null) await _changeMode(picked);
  }

  Future<void> _changeMode(DetectionMode next) async {
    if (next == _sample.detectionMode || _sample.id == null) return;
    final updated = await SampleRepository.instance
        .updateDetectionMode(sampleId: _sample.id!, mode: next);
    if (!mounted || updated == null) return;
    setState(() {
      _sample = updated;
      _error = null;
    });
    // Re-run the appropriate pipeline with the new analyzer so existing
    // instances get re-classified for the new mode (or, for the first switch
    // into a manual-segmentation mode, the working image just stays put).
    DetectionService.instance.requeue(
      sampleId: updated.id!,
      filePath: updated.filePath,
      mode: updated.detectionMode,
    );
  }

  /// Refresh both overlays for the whole sample after the user edits, adds,
  /// or deletes an instance — we rebuild the segmentation overlay from the
  /// new masks and re-run only the disease step (not the full detection
  /// pipeline) since the masks themselves came from the editor.
  Future<void> _refreshAfterEdit() async {
    if (_sample.id == null ||
        _workingPng == null ||
        _workingW == null ||
        _workingH == null) {
      return;
    }
    try {
      final segPng =
          await DetectionService.instance.rebuildSegmentationOverlay(
        sampleId: _sample.id!,
        workingW: _workingW!,
        workingH: _workingH!,
        instances: _instances,
      );
      if (!mounted) return;
      setState(() => _segmentationOverlayPng = segPng);

      if (_instances.isEmpty) {
        await SampleRepository.instance
            .saveDiseaseOverlay(sampleId: _sample.id!, png: null);
        if (!mounted) return;
        setState(() => _diseaseOverlayPng = null);
        return;
      }
      final analyzed = await DetectionService.instance.runDiseaseAnalysis(
        mode: _sample.detectionMode,
        workingPng: _workingPng!,
        workingW: _workingW!,
        workingH: _workingH!,
        instances: List<SampleInstance>.from(_instances),
        sampleId: _sample.id!,
      );
      if (!mounted) return;
      setState(() {
        _instances = analyzed.instances;
        _diseaseOverlayPng = analyzed.overlayPng;
      });
    } catch (e, st) {
      debugPrint('[refresh] post-edit refresh failed: $e\n$st');
    }
  }

  Future<void> _openInstance(SampleInstance instance) async {
    if (_workingPng == null || _workingW == null || _workingH == null) return;
    final indexLabel = _instances.indexOf(instance) + 1;
    final result = await Navigator.of(context).push<InspectorResult>(
      MaterialPageRoute(
        builder: (_) => InstanceInspectorScreen(
          sample: _sample,
          instance: instance,
          indexLabel: indexLabel,
          workingPng: _workingPng!,
          workingWidth: _workingW!,
          workingHeight: _workingH!,
        ),
      ),
    );
    if (result == null) return;
    final editorResult = result.editorResult;
    final updatedSample = result.sample;
    setState(() {
      if (updatedSample != null) _sample = updatedSample;
      if (editorResult != null) {
        if (editorResult.deleted) {
          _instances.removeWhere((it) => it.id == editorResult.instance.id);
        } else {
          final i = _instances
              .indexWhere((it) => it.id == editorResult.instance.id);
          if (i >= 0) _instances[i] = editorResult.instance;
        }
      }
    });
    // Mask changed — re-run FHB across the sample so the toggle/chart agree.
    // Skip when only the sample tag was edited; that doesn't affect masks.
    if (editorResult != null) {
      await _refreshAfterEdit();
    }
  }

  Future<void> _addInstance() async {
    if (_workingPng == null ||
        _workingW == null ||
        _workingH == null ||
        _sample.id == null) {
      return;
    }
    final result = await Navigator.of(context).push<InstanceEditorResult>(
      MaterialPageRoute(
        builder: (_) => InstanceEditorScreen.create(
          sampleId: _sample.id!,
          workingPng: _workingPng!,
          workingWidth: _workingW!,
          workingHeight: _workingH!,
        ),
      ),
    );
    if (result == null || result.deleted) return;
    setState(() => _instances.add(result.instance));
    await _refreshAfterEdit();
  }

  Future<void> _confirmDeleteInstance(SampleInstance instance) async {
    if (instance.id == null) return;
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: const Text('Delete instance?',
            style: TextStyle(color: Colors.white)),
        content: const Text(
          'This will remove the segmentation mask for this instance.',
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
    await SampleRepository.instance.deleteInstance(instance.id!);
    if (!mounted) return;
    setState(() => _instances.removeWhere((it) => it.id == instance.id));
    await _refreshAfterEdit();
  }

  bool get _hasDiseaseData =>
      _instances.any((i) => i.hasDiseaseAnalysis);

  bool get _hasInstances => _instances.isNotEmpty;

  /// Pick the right overlay layer for the current mode. Both PNG-based
  /// overlays (segmentation, disease) are at working-image resolution, so
  /// they're stretched to fit the displayed original via `BoxFit.fill`. The
  /// bbox layer paints into whatever pixel size the parent gives it.
  Widget _buildOverlay({
    required bool inDiseaseMode,
    required bool inSegmentationMode,
  }) {
    if (inDiseaseMode && _diseaseOverlayPng != null) {
      return Image.memory(
        _diseaseOverlayPng!,
        gaplessPlayback: true,
        fit: BoxFit.fill,
      );
    }
    if (inSegmentationMode && _segmentationOverlayPng != null) {
      return Image.memory(
        _segmentationOverlayPng!,
        gaplessPlayback: true,
        fit: BoxFit.fill,
      );
    }
    // Default / fallback: bbox + centroid lines.
    return _BboxOverlay(
      instances: _instances,
      width: _workingW ?? 1,
      height: _workingH ?? 1,
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasWorking = _workingPng != null;
    final aspect =
        hasWorking ? (_workingW! / _workingH!) : 1.0; // safe fallback
    final inDiseaseMode =
        _overlayMode == _OverlayMode.disease && _hasDiseaseData;
    final inSegmentationMode =
        _overlayMode == _OverlayMode.segmentation && _hasInstances;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Text(
          _formatDate(_sample.takenAt),
          style: const TextStyle(color: Colors.white, fontSize: 15),
        ),
        actions: [
          _ViewerModeChip(mode: _sample.detectionMode, onPick: _pickMode),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Stack(
                children: [
                  Positioned.fill(
                    child: Center(
                      child: InteractiveViewer(
                        minScale: 1,
                        maxScale: 5,
                        child: hasWorking
                            ? AspectRatio(
                                aspectRatio: aspect,
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    // Always show the original capture as
                                    // the zoomable background. Overlays
                                    // sit on top and are stretched via
                                    // BoxFit.fill since the working image
                                    // shares its aspect ratio.
                                    Image.file(
                                      File(_sample.filePath),
                                      gaplessPlayback: true,
                                      fit: BoxFit.fill,
                                      cacheWidth: 2048,
                                      errorBuilder: (_, __, ___) =>
                                          const Icon(
                                        Icons.broken_image_outlined,
                                        color: Colors.white54,
                                        size: 64,
                                      ),
                                    ),
                                    if (_showOverlay && _hasInstances)
                                      _buildOverlay(
                                        inDiseaseMode: inDiseaseMode,
                                        inSegmentationMode:
                                            inSegmentationMode,
                                      ),
                                  ],
                                ),
                              )
                            : Image.file(
                                File(_sample.filePath),
                                cacheWidth: 2048,
                                errorBuilder: (_, __, ___) => const Icon(
                                  Icons.broken_image_outlined,
                                  color: Colors.white54,
                                  size: 64,
                                ),
                              ),
                      ),
                    ),
                  ),
                  if (_lastElapsed != null && _instances.isNotEmpty)
                    Positioned(
                      left: 16,
                      right: 16,
                      bottom: 12,
                      child: _StatusPill(
                        text: '${_instances.length} '
                            '${_instances.length == 1 ? _sample.detectionMode.instanceNounSingular : _sample.detectionMode.instanceNounPlural}'
                            ' · ${_lastElapsed!.inSeconds} s',
                      ),
                    ),
                  if (_error != null)
                    Positioned(
                      left: 16,
                      right: 16,
                      bottom: 12,
                      child: Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.redAccent.withValues(alpha: 0.8),
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
            if (_hasInstances) ...[
              _MaskVisibilityToggle(
                show: _showOverlay,
                onChanged: (v) => setState(() => _showOverlay = v),
              ),
              if (_showOverlay)
                _OverlayModeToggle(
                  mode: _overlayMode,
                  hasDisease: _hasDiseaseData,
                  onChanged: (m) => setState(() => _overlayMode = m),
                ),
            ],
            _CollectionPanel(
              collection: _collection,
              onTap: _changeCollection,
            ),
            _SampleTagPanel(sample: _sample, onTap: _editSampleTag),
            _InstanceStrip(
              instances: _instances,
              canAdd: hasWorking,
              showDisease: inDiseaseMode,
              mode: _sample.detectionMode,
              onTap: _openInstance,
              onLongPress: _confirmDeleteInstance,
              onAdd: _addInstance,
            ),
            // The distribution chart bins by FHB ratio, which is meaningful
            // only for the wheat HSV pipeline. Grape-leaf samples carry a
            // SwinV2 disease label, not a continuous ratio.
            if (_hasDiseaseData &&
                _sample.detectionMode == DetectionMode.wheatFhb)
              _FhbDistributionChart(
                instances: _instances,
                title: _sample.detectionMode.distributionChartTitle,
              ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: SizedBox(
                width: double.infinity,
                child: _DetectButton(
                  status: _jobStatus,
                  hasInstances: _instances.isNotEmpty,
                  mode: _sample.detectionMode,
                  onPressed: _detect,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Compact one-line indicator on a samples-list row showing the live status
/// of the sample's background detection job: queue/progress while running,
/// "n wheat heads" / "n leaves" when done, error text when failed, and
/// nothing when the pipeline is idle (e.g. for legacy samples that pre-date
/// auto-detection).
class _SampleJobStatusRow extends StatelessWidget {
  const _SampleJobStatusRow({required this.sampleId, required this.mode});

  final int sampleId;
  final DetectionMode mode;

  @override
  Widget build(BuildContext context) {
    final listenable = DetectionService.instance.statusFor(sampleId);
    return ValueListenableBuilder<DetectionStatus>(
      valueListenable: listenable,
      builder: (_, status, __) {
        if (status is DetectionRunning) {
          return Row(
            children: [
              const SizedBox(
                width: 12,
                height: 12,
                child: CircularProgressIndicator(
                  strokeWidth: 1.6,
                  color: Colors.white70,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      status.phase,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                    if (status.progress != null) ...[
                      const SizedBox(height: 3),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(2),
                        child: LinearProgressIndicator(
                          value: status.progress,
                          minHeight: 3,
                          backgroundColor: Colors.white12,
                          valueColor: const AlwaysStoppedAnimation<Color>(
                            Colors.white70,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          );
        }
        if (status is DetectionCompleted) {
          final noun = status.detectionCount == 1
              ? mode.instanceNounSingular
              : mode.instanceNounPlural;
          return Row(
            children: [
              const Icon(Icons.check_circle_outline,
                  size: 14, color: Colors.greenAccent),
              const SizedBox(width: 4),
              Text(
                '${status.detectionCount} $noun '
                '· ${status.elapsed.inMilliseconds} ms',
                style: const TextStyle(
                    color: Colors.greenAccent, fontSize: 12),
              ),
            ],
          );
        }
        if (status is DetectionFailed) {
          return const Row(
            children: [
              Icon(Icons.error_outline, size: 14, color: Colors.redAccent),
              SizedBox(width: 4),
              Expanded(
                child: Text(
                  'Detection failed',
                  style: TextStyle(color: Colors.redAccent, fontSize: 12),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          );
        }
        return const SizedBox.shrink();
      },
    );
  }
}

class _DetectButton extends StatelessWidget {
  const _DetectButton({
    required this.status,
    required this.hasInstances,
    required this.mode,
    required this.onPressed,
  });

  /// Listenable for the sample's detection job. Null when the sample has no
  /// id yet (which shouldn't happen here, but defends the UI either way).
  final ValueListenable<DetectionStatus>? status;
  final bool hasInstances;
  final DetectionMode mode;
  final VoidCallback onPressed;

  /// Label depends on mode + whether instances already exist:
  ///
  /// - Auto-detect mode (FHB): `Detect wheat heads` / `Re-run detection`.
  /// - Manual mode (Grape Leaf): `Process image` / `Re-analyze leaves` —
  ///   "detect" would be misleading since the user paints leaves themselves.
  String get _idleLabel {
    if (mode.hasAutoDetection) {
      return hasInstances
          ? 'Re-run detection'
          : 'Detect ${mode.instanceNounPlural}';
    }
    return hasInstances
        ? 'Re-analyze ${mode.instanceNounPlural}'
        : 'Process image';
  }

  @override
  Widget build(BuildContext context) {
    if (status == null) {
      return _buildIdleButton(label: _idleLabel);
    }
    return ValueListenableBuilder<DetectionStatus>(
      valueListenable: status!,
      builder: (_, value, __) {
        if (value is DetectionRunning) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            mainAxisSize: MainAxisSize.min,
            children: [
              FilledButton.icon(
                style: FilledButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                ),
                onPressed: null,
                icon: const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Colors.black,
                  ),
                ),
                label: Text(value.phase),
              ),
              if (value.progress != null) ...[
                const SizedBox(height: 6),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: value.progress,
                    minHeight: 4,
                    backgroundColor: Colors.white12,
                    valueColor:
                        const AlwaysStoppedAnimation<Color>(Colors.white),
                  ),
                ),
              ],
            ],
          );
        }
        return _buildIdleButton(label: _idleLabel);
      },
    );
  }

  Widget _buildIdleButton({required String label}) {
    return FilledButton.icon(
      style: FilledButton.styleFrom(
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
      ),
      onPressed: onPressed,
      icon: const Icon(Icons.auto_awesome),
      label: Text(label),
    );
  }
}

class _ViewerModeChip extends StatelessWidget {
  const _ViewerModeChip({required this.mode, required this.onPick});

  final DetectionMode mode;
  final VoidCallback onPick;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
      child: GestureDetector(
        onTap: onPick,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: Colors.white12,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: Colors.white24, width: 1),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                mode.label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(width: 4),
              const Icon(Icons.expand_more, size: 14, color: Colors.white70),
            ],
          ),
        ),
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        text,
        style: const TextStyle(color: Colors.white, fontSize: 13),
        textAlign: TextAlign.center,
      ),
    );
  }
}

/// Lightweight bbox + centroid overlay (used in "Bbox" mode). Painted with
/// CustomPaint so it scales smoothly with the viewer's zoom regardless of
/// the underlying image's pixel resolution. Bboxes are in working-image
/// coords; the painter divides by the working dims to map them onto the
/// widget's render size, which spans the whole displayed image.
class _BboxOverlay extends StatelessWidget {
  const _BboxOverlay({
    required this.instances,
    required this.width,
    required this.height,
  });

  final List<SampleInstance> instances;
  final int width;
  final int height;

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: CustomPaint(
        painter: _BboxPainter(
          instances: instances,
          imgW: width,
          imgH: height,
        ),
      ),
    );
  }
}

class _BboxPainter extends CustomPainter {
  _BboxPainter({
    required this.instances,
    required this.imgW,
    required this.imgH,
  });

  final List<SampleInstance> instances;
  final int imgW;
  final int imgH;

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = Paint()
      ..color = const Color.fromARGB(220, 0, 230, 118)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final dot = Paint()
      ..color = const Color.fromARGB(255, 255, 64, 129)
      ..style = PaintingStyle.fill;
    final sx = size.width / imgW;
    final sy = size.height / imgH;
    for (final inst in instances) {
      final r = Rect.fromLTRB(
        inst.bbox.left * sx,
        inst.bbox.top * sy,
        inst.bbox.right * sx,
        inst.bbox.bottom * sy,
      );
      canvas.drawRect(r, stroke);
      canvas.drawCircle(
        Offset(inst.centroid.dx * sx, inst.centroid.dy * sy),
        4,
        dot,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _BboxPainter old) =>
      old.instances != instances || old.imgW != imgW || old.imgH != imgH;
}

/// Compact info panel rendered on the sample viewer when the sample carries
/// QR-tag metadata. Mirrors the JSON payload (ID, Line, Rep, Location, Note)
/// scanned at capture time so users can correlate the on-screen sample with
/// the physical plot tag without re-scanning the code.
/// Sample-viewer panel showing the sample's current collection (or
/// "Uncollected") with a "Change" affordance. Tap anywhere on the row to
/// open the picker; mirrors the QR sample-tag panel's visual treatment.
class _CollectionPanel extends StatelessWidget {
  const _CollectionPanel({required this.collection, required this.onTap});

  final Collection? collection;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final hasCollection = collection != null;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(10),
        child: Container(
          width: double.infinity,
          padding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.06),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: hasCollection ? Colors.amberAccent : Colors.white24,
              width: hasCollection ? 1.2 : 1,
            ),
          ),
          child: Row(
            children: [
              Icon(
                Icons.collections_bookmark_outlined,
                size: 16,
                color: hasCollection
                    ? Colors.amberAccent
                    : Colors.white54,
              ),
              const SizedBox(width: 8),
              const Text(
                'COLLECTION',
                style: TextStyle(
                  color: Colors.white60,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.8,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  hasCollection ? collection!.name : 'Uncollected',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: hasCollection ? Colors.white : Colors.white54,
                    fontSize: 13,
                    fontWeight: hasCollection
                        ? FontWeight.w600
                        : FontWeight.normal,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              const Icon(Icons.edit, size: 14, color: Colors.white54),
            ],
          ),
        ),
      ),
    );
  }
}

class _SampleTagPanel extends StatelessWidget {
  const _SampleTagPanel({required this.sample, required this.onTap});

  final Sample sample;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final rows = <_TagRow>[];
    void addRow(String label, String? value) {
      if (value == null || value.isEmpty) return;
      rows.add(_TagRow(label: label, value: value));
    }

    addRow('ID', sample.qrId);
    addRow('Line', sample.qrLine);
    addRow('Rep', sample.qrRep);
    addRow('Location', sample.qrLocation);
    addRow('Note', sample.qrNote);

    final hasTag = rows.isNotEmpty;

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(10),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.06),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: hasTag ? Colors.lightBlueAccent : Colors.white24,
              width: hasTag ? 1.2 : 1,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.qr_code,
                    color: hasTag ? Colors.lightBlueAccent : Colors.white54,
                    size: 16,
                  ),
                  const SizedBox(width: 6),
                  const Text(
                    'SAMPLE TAG',
                    style: TextStyle(
                      color: Colors.white60,
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.8,
                    ),
                  ),
                  const Spacer(),
                  const Icon(Icons.edit, size: 14, color: Colors.white54),
                ],
              ),
              const SizedBox(height: 8),
              if (hasTag)
                for (final r in rows) r
              else
                const Text(
                  'No tag attached. Tap to add an ID, line, rep, location, '
                  'or note.',
                  style: TextStyle(color: Colors.white54, fontSize: 13),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _TagRow extends StatelessWidget {
  const _TagRow({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 72,
            child: Text(
              label,
              style: const TextStyle(color: Colors.white54, fontSize: 12),
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

class _InstanceStrip extends StatelessWidget {
  const _InstanceStrip({
    required this.instances,
    required this.canAdd,
    required this.showDisease,
    required this.mode,
    required this.onTap,
    required this.onLongPress,
    required this.onAdd,
  });

  final List<SampleInstance> instances;

  /// Show the leading "+" tile only once a working image is available
  /// (no point creating an instance before detection has run).
  final bool canAdd;

  /// When true, swap the segmentation preview tile for the per-pixel disease
  /// classification preview (falls back to the segmentation preview for
  /// instances that haven't been analysed yet).
  final bool showDisease;

  /// Detection mode for this sample. Drives the per-tile caption: wheat
  /// shows an FHB%, grape shows the SwinV2 disease class.
  final DetectionMode mode;
  final ValueChanged<SampleInstance> onTap;
  final ValueChanged<SampleInstance> onLongPress;
  final VoidCallback onAdd;

  String? _captionFor(SampleInstance inst) {
    switch (mode) {
      case DetectionMode.wheatFhb:
        return inst.fhbRatio != null
            ? '${(inst.fhbRatio! * 100).toStringAsFixed(0)}%'
            : null;
      case DetectionMode.grapeLeaf:
        final severity = inst.fhbSeverity;
        return severity == null ? null : grapeLeafDisplayLabel(severity);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (instances.isEmpty && !canAdd) return const SizedBox.shrink();
    final addTileCount = canAdd ? 1 : 0;
    final total = instances.length + addTileCount;
    return Container(
      height: 124,
      padding: const EdgeInsets.symmetric(vertical: 8),
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: Colors.white12)),
      ),
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        itemCount: total,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          if (canAdd && i == 0) return _AddInstanceTile(onTap: onAdd);
          final inst = instances[i - addTileCount];
          final tile = (showDisease && inst.diseasePreviewPng != null)
              ? inst.diseasePreviewPng!
              : inst.previewPng;
          final caption = _captionFor(inst);
          return GestureDetector(
            onTap: () => onTap(inst),
            onLongPress: () => onLongPress(inst),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.memory(
                    tile,
                    width: 80,
                    height: 80,
                    fit: BoxFit.cover,
                    gaplessPlayback: true,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '#${i - addTileCount + 1}'
                  '${caption != null ? ' · $caption' : ''}',
                  style: const TextStyle(color: Colors.white70, fontSize: 11),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _MaskVisibilityToggle extends StatelessWidget {
  const _MaskVisibilityToggle({required this.show, required this.onChanged});

  final bool show;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
      child: Row(
        children: [
          Icon(
            show ? Icons.visibility : Icons.visibility_off,
            size: 18,
            color: Colors.white70,
          ),
          const SizedBox(width: 8),
          const Text(
            'Show masks',
            style: TextStyle(color: Colors.white, fontSize: 13),
          ),
          const Spacer(),
          Switch(
            value: show,
            onChanged: onChanged,
            activeThumbColor: Colors.white,
            activeTrackColor: Colors.white24,
            inactiveThumbColor: Colors.white54,
            inactiveTrackColor: Colors.white12,
          ),
        ],
      ),
    );
  }
}

class _OverlayModeToggle extends StatelessWidget {
  const _OverlayModeToggle({
    required this.mode,
    required this.hasDisease,
    required this.onChanged,
  });

  final _OverlayMode mode;

  /// Disable the disease segment until FHB analysis has produced data —
  /// otherwise tapping it would show an empty overlay.
  final bool hasDisease;
  final ValueChanged<_OverlayMode> onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: SegmentedButton<_OverlayMode>(
        showSelectedIcon: false,
        segments: [
          const ButtonSegment(
            value: _OverlayMode.bbox,
            label: Text('Bbox'),
            icon: Icon(Icons.crop_square),
          ),
          const ButtonSegment(
            value: _OverlayMode.segmentation,
            label: Text('Segment'),
            icon: Icon(Icons.gesture),
          ),
          ButtonSegment(
            value: _OverlayMode.disease,
            label: const Text('Disease'),
            icon: const Icon(Icons.local_florist),
            enabled: hasDisease,
          ),
        ],
        selected: {mode},
        onSelectionChanged: (s) => onChanged(s.first),
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
      ),
    );
  }
}

/// Histogram of per-instance FHB% (necrotic / (necrotic + green)) across the
/// sample. 10 bins, each 10% wide. Mirrors the per-spike summary chart in
/// the FHB notebook, condensed to a single row that can sit under the strip.
class _FhbDistributionChart extends StatelessWidget {
  const _FhbDistributionChart({
    required this.instances,
    required this.title,
  });

  final List<SampleInstance> instances;
  final String title;

  static const int _kBinCount = 10;

  @override
  Widget build(BuildContext context) {
    final analysed = instances.where((i) => i.hasDiseaseAnalysis).toList();
    if (analysed.isEmpty) return const SizedBox.shrink();
    final bins = List<int>.filled(_kBinCount, 0);
    for (final inst in analysed) {
      final pct = (inst.fhbRatio ?? 0).clamp(0.0, 1.0);
      var idx = (pct * _kBinCount).floor();
      if (idx >= _kBinCount) idx = _kBinCount - 1;
      bins[idx] += 1;
    }
    final maxCount = bins.reduce((a, b) => a > b ? a : b);
    final mean = analysed.map((i) => i.fhbRatio ?? 0).reduce((a, b) => a + b) /
        analysed.length;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: Colors.white12)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              Text(
                'mean ${(mean * 100).toStringAsFixed(0)}% · '
                'n=${analysed.length}',
                style: const TextStyle(color: Colors.white60, fontSize: 11),
              ),
            ],
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 60,
            child: CustomPaint(
              painter: _FhbHistogramPainter(bins: bins, maxCount: maxCount),
              size: Size.infinite,
            ),
          ),
          const SizedBox(height: 4),
          const Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('0%',
                  style: TextStyle(color: Colors.white54, fontSize: 10)),
              Text('50%',
                  style: TextStyle(color: Colors.white54, fontSize: 10)),
              Text('100%',
                  style: TextStyle(color: Colors.white54, fontSize: 10)),
            ],
          ),
        ],
      ),
    );
  }
}

class _FhbHistogramPainter extends CustomPainter {
  _FhbHistogramPainter({required this.bins, required this.maxCount});

  final List<int> bins;
  final int maxCount;

  @override
  void paint(Canvas canvas, Size size) {
    final n = bins.length;
    const gap = 2.0;
    final barWidth = (size.width - gap * (n - 1)) / n;
    final scale = maxCount == 0 ? 0.0 : size.height / maxCount;
    for (int i = 0; i < n; i++) {
      final count = bins[i];
      final h = (count * scale).clamp(0.0, size.height);
      final x = i * (barWidth + gap);
      final rect = Rect.fromLTWH(x, size.height - h, barWidth, h);
      // Tint walks from healthy-green (left) to FHB-red (right).
      final t = i / (n - 1);
      final color = Color.lerp(
        const Color(0xFF00C853),
        const Color(0xFFE53935),
        t,
      )!;
      final paint = Paint()..color = color.withValues(alpha: 0.85);
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(2)),
        paint,
      );
      if (count > 0) {
        final tp = TextPainter(
          text: TextSpan(
            text: '$count',
            style: const TextStyle(color: Colors.white, fontSize: 9),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        final tx = x + (barWidth - tp.width) / 2;
        final ty = (size.height - h - tp.height - 1).clamp(0.0, size.height);
        tp.paint(canvas, Offset(tx, ty));
      }
    }
    // Subtle baseline.
    final baseline = Paint()
      ..color = Colors.white12
      ..strokeWidth = 1;
    canvas.drawLine(
      Offset(0, size.height - 0.5),
      Offset(size.width, size.height - 0.5),
      baseline,
    );
  }

  @override
  bool shouldRepaint(covariant _FhbHistogramPainter old) =>
      old.bins != bins || old.maxCount != maxCount;
}

class _AddInstanceTile extends StatelessWidget {
  const _AddInstanceTile({required this.onTap});
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: Colors.white10,
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: Colors.white24, style: BorderStyle.solid),
            ),
            child: const Icon(Icons.add, color: Colors.white70, size: 28),
          ),
          const SizedBox(height: 4),
          const Text(
            'New',
            style: TextStyle(color: Colors.white70, fontSize: 11),
          ),
        ],
      ),
    );
  }
}

String _formatDate(DateTime dt) {
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

String _qrLabel(Sample sample) {
  final parts = <String>[];
  if (sample.qrId != null && sample.qrId!.isNotEmpty) {
    parts.add(sample.qrId!);
  }
  if (sample.qrLine != null && sample.qrLine!.isNotEmpty) {
    parts.add(sample.qrLine!);
  }
  if (sample.qrRep != null && sample.qrRep!.isNotEmpty) {
    parts.add('rep ${sample.qrRep!}');
  }
  if (parts.isEmpty &&
      sample.qrLocation != null &&
      sample.qrLocation!.isNotEmpty) {
    return sample.qrLocation!;
  }
  if (parts.isEmpty && sample.qrNote != null && sample.qrNote!.isNotEmpty) {
    return sample.qrNote!;
  }
  return parts.join(' · ');
}
