import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:gopher_eye/model/collection.dart';
import 'package:gopher_eye/model/detection_mode.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/model/sample_instance.dart';
import 'package:gopher_eye/screens/collection_picker_screen.dart';
import 'package:gopher_eye/screens/export_screen.dart';
import 'package:gopher_eye/screens/instance_editor_screen.dart';
import 'package:gopher_eye/screens/instance_inspector_screen.dart';
import 'package:gopher_eye/services/detection_service.dart';
import 'package:gopher_eye/services/grape_leaf_pipeline.dart';
import 'package:gopher_eye/services/sample_repository.dart';
import 'package:gopher_eye/widgets/sample_tag_edit_dialog.dart';

class SamplesScreen extends StatefulWidget {
  const SamplesScreen({super.key, this.onBack});

  final VoidCallback? onBack;

  @override
  State<SamplesScreen> createState() => _SamplesScreenState();
}

/// Sentinel used as the "collection id" of the uncollected bucket so the
/// expand-state set and the samples-by-collection map can share one key
/// space without juggling `null` checks at every call site.
const int _kUncollectedKey = -1;

/// A collection plus its samples, prepared for rendering as one expandable
/// section in the grouped samples list. The same shape is used for the
/// synthetic "Uncollected" bucket — its [collection] is `null` and its id
/// is [_kUncollectedKey].
class _CollectionGroup {
  _CollectionGroup({
    required this.id,
    required this.collection,
    required this.samples,
  });

  final int id;
  final Collection? collection;
  final List<Sample> samples;

  bool get isUncollected => collection == null;
  String get displayName => collection?.name ?? 'Uncollected';

  /// Newest sample's timestamp (groups are sorted by this so the most
  /// recently active collection floats to the top). Falls back to the
  /// collection's createdAt for empty collections.
  DateTime get sortKey => samples.isNotEmpty
      ? samples.first.takenAt
      : (collection?.createdAt ?? DateTime.fromMillisecondsSinceEpoch(0));

  String? get coverImagePath =>
      samples.isNotEmpty ? samples.first.filePath : null;
}

class _SamplesScreenState extends State<SamplesScreen> {
  late Future<void> _future;
  List<_CollectionGroup> _groups = [];

  /// IDs of collections (and the [_kUncollectedKey] bucket) the user has
  /// expanded. Persisted across reloads so swipe-deletes don't snap a
  /// section closed.
  final Set<int> _expanded = {};

  final TextEditingController _searchCtrl = TextEditingController();
  String _query = '';

  @override
  void initState() {
    super.initState();
    _searchCtrl.addListener(_onSearchChanged);
    _future = _load();
    DetectionService.instance.addListener(_onAnyJobCompleted);
  }

  @override
  void dispose() {
    _searchCtrl.removeListener(_onSearchChanged);
    _searchCtrl.dispose();
    DetectionService.instance.removeListener(_onAnyJobCompleted);
    super.dispose();
  }

  void _onSearchChanged() {
    if (_searchCtrl.text == _query) return;
    setState(() => _query = _searchCtrl.text);
  }

  void _onAnyJobCompleted() {
    // The service notifies listeners when a job completes/fails so the list
    // can pick up newly-saved instances. The per-tile status row already
    // updates itself via its own ValueListenable; this hook is just a nudge
    // for any list-level state that might depend on completion.
    if (mounted) setState(() {});
  }

  Future<void> _load() async {
    try {
      final repo = SampleRepository.instance;
      final samples = await repo.listAll();
      final collections = await repo.listCollections();
      debugPrint(
        '[samples] loaded ${samples.length} samples, '
        '${collections.length} collections',
      );

      // Bucket samples by collection_id (using the sentinel for the
      // uncollected pile so a single map can hold both kinds).
      final byCollection = <int, List<Sample>>{};
      for (final s in samples) {
        final key = s.collectionId ?? _kUncollectedKey;
        (byCollection[key] ??= []).add(s);
      }
      // Newest sample first in each bucket — drives both the cover thumb
      // and the in-section ordering.
      for (final list in byCollection.values) {
        list.sort((a, b) => b.takenAt.compareTo(a.takenAt));
      }

      final groups = <_CollectionGroup>[
        for (final c in collections)
          _CollectionGroup(
            id: c.id!,
            collection: c,
            samples: byCollection[c.id] ?? const [],
          ),
        if ((byCollection[_kUncollectedKey] ?? const []).isNotEmpty)
          _CollectionGroup(
            id: _kUncollectedKey,
            collection: null,
            samples: byCollection[_kUncollectedKey]!,
          ),
      ]..sort((a, b) => b.sortKey.compareTo(a.sortKey));

      if (mounted) setState(() => _groups = groups);
    } catch (e, st) {
      debugPrint('[samples] load failed: $e\n$st');
      rethrow;
    }
  }

  void _refresh() {
    setState(() {
      _groups = [];
      _future = _load();
    });
  }

  Future<void> _delete(Sample sample) async {
    if (sample.id == null) return;
    await SampleRepository.instance.delete(sample.id!);
    DetectionService.instance.forget(sample.id!);
    if (!mounted) return;
    setState(() {
      for (final g in _groups) {
        g.samples.removeWhere((s) => s.id == sample.id);
      }
    });
  }

  void _toggleExpanded(int id) {
    setState(() {
      if (!_expanded.remove(id)) _expanded.add(id);
    });
  }

  /// Does this sample match the current query? Checks date string + every
  /// QR-tag field so the search bar covers "find the rep 3 sample" /
  /// "find anything tagged with ID-42" use cases without a dedicated UI.
  bool _sampleMatchesQuery(Sample s, String query) {
    if (query.isEmpty) return true;
    final q = query.toLowerCase();
    if (_formatDate(s.takenAt).toLowerCase().contains(q)) return true;
    for (final field in [
      s.qrId,
      s.qrLine,
      s.qrRep,
      s.qrLocation,
      s.qrNote,
    ]) {
      if (field != null && field.toLowerCase().contains(q)) return true;
    }
    return false;
  }

  /// Build the list of groups to render given the active search query. A
  /// group passes when its name matches OR at least one of its samples
  /// matches; non-matching samples inside an otherwise-matching collection
  /// are filtered out so the user sees only relevant rows.
  List<_CollectionGroup> _visibleGroups() {
    final q = _query.trim().toLowerCase();
    if (q.isEmpty) return _groups;
    final out = <_CollectionGroup>[];
    for (final g in _groups) {
      final nameMatches = g.displayName.toLowerCase().contains(q);
      if (nameMatches) {
        out.add(g);
        continue;
      }
      final matchingSamples =
          g.samples.where((s) => _sampleMatchesQuery(s, q)).toList();
      if (matchingSamples.isNotEmpty) {
        out.add(_CollectionGroup(
          id: g.id,
          collection: g.collection,
          samples: matchingSamples,
        ));
      }
    }
    return out;
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

  Future<void> _openViewer(Sample sample) async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => SampleViewerScreen(sample: sample)),
    );
    // The viewer can reassign a sample to a different collection (or clear
    // it). Reload so the section a sample lives under reflects the change.
    if (mounted) _refresh();
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
      body: SafeArea(
        child: Column(
          children: [
            _CollectionSearchField(
              controller: _searchCtrl,
              onClear: () => _searchCtrl.clear(),
            ),
            Expanded(
              child: FutureBuilder<void>(
                future: _future,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting &&
                      _groups.isEmpty) {
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
                  if (_groups.isEmpty) {
                    return const Center(
                      child: Text(
                        'No samples yet — capture some plants',
                        style: TextStyle(color: Colors.white70),
                      ),
                    );
                  }
                  final visible = _visibleGroups();
                  if (visible.isEmpty) {
                    return const Center(
                      child: Text(
                        'No collections or samples match',
                        style: TextStyle(color: Colors.white54),
                      ),
                    );
                  }
                  // When searching, auto-expand any group with matching
                  // samples so results are visible without an extra tap.
                  final searching = _query.trim().isNotEmpty;
                  return ListView.builder(
                    itemCount: visible.length,
                    itemBuilder: (context, index) {
                      final group = visible[index];
                      final expanded =
                          searching || _expanded.contains(group.id);
                      return _CollectionSection(
                        group: group,
                        expanded: expanded,
                        onToggle: () => _toggleExpanded(group.id),
                        onOpenSample: _openViewer,
                        onConfirmDelete: _confirmDelete,
                        onDelete: _delete,
                      );
                    },
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Search box pinned above the samples list. Matches against collection
/// names and sample QR-tag fields; see [_SamplesScreenState._sampleMatchesQuery].
class _CollectionSearchField extends StatelessWidget {
  const _CollectionSearchField({
    required this.controller,
    required this.onClear,
  });

  final TextEditingController controller;
  final VoidCallback onClear;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
      child: TextField(
        controller: controller,
        style: const TextStyle(color: Colors.white),
        decoration: InputDecoration(
          hintText: 'Search collections or samples',
          hintStyle: const TextStyle(color: Colors.white38),
          prefixIcon: const Icon(Icons.search, color: Colors.white54),
          suffixIcon: ValueListenableBuilder<TextEditingValue>(
            valueListenable: controller,
            builder: (_, value, __) {
              if (value.text.isEmpty) return const SizedBox.shrink();
              return IconButton(
                icon: const Icon(Icons.close, color: Colors.white54),
                onPressed: onClear,
              );
            },
          ),
          filled: true,
          fillColor: Colors.white10,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: BorderSide.none,
          ),
        ),
      ),
    );
  }
}

/// One expandable section in the grouped samples list: a [CollectionTile]
/// header that toggles open/closed, and (when open) the section's [SampleTile]
/// rows beneath it. Tap the header to expand/collapse; long-press inside is
/// reserved for sample-level actions.
class _CollectionSection extends StatelessWidget {
  const _CollectionSection({
    required this.group,
    required this.expanded,
    required this.onToggle,
    required this.onOpenSample,
    required this.onConfirmDelete,
    required this.onDelete,
  });

  final _CollectionGroup group;
  final bool expanded;
  final VoidCallback onToggle;
  final void Function(Sample) onOpenSample;
  final Future<bool> Function(Sample) onConfirmDelete;
  final Future<void> Function(Sample) onDelete;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Divider(height: 1, color: Colors.white12),
        CollectionTile(
          collection: group.collection,
          isUncollected: group.isUncollected,
          sampleCount: group.samples.length,
          lastSampleAt: group.sortKey,
          coverImagePath: group.coverImagePath,
          expanded: expanded,
          onTap: onToggle,
        ),
        if (expanded)
          for (final sample in group.samples) ...[
            const Divider(height: 1, color: Colors.white12),
            Dismissible(
              key: ValueKey('sample-${sample.id}'),
              direction: DismissDirection.endToStart,
              background: Container(
                color: Colors.redAccent,
                alignment: Alignment.centerRight,
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: const Icon(Icons.delete, color: Colors.white),
              ),
              confirmDismiss: (_) => onConfirmDelete(sample),
              onDismissed: (_) => onDelete(sample),
              // Slight inset on the sample tile so the hierarchy reads
              // "samples belong to the collection above."
              child: Container(
                padding: const EdgeInsets.only(left: 12),
                color: Colors.white.withValues(alpha: 0.02),
                child: SampleTile(
                  sample: sample,
                  onTap: () => onOpenSample(sample),
                  onLongPress: () async {
                    if (await onConfirmDelete(sample)) await onDelete(sample);
                  },
                ),
              ),
            ),
          ],
      ],
    );
  }
}

/// Header row for a collection (or the synthetic "Uncollected" group) on
/// the samples scroll page. Shows the cover thumbnail (latest sample image,
/// or a placeholder when empty), name, sample count, the timestamp of the
/// latest capture, and a chevron that rotates to indicate expanded state.
/// Tap toggles the section open/closed.
class CollectionTile extends StatelessWidget {
  const CollectionTile({
    super.key,
    required this.collection,
    required this.sampleCount,
    required this.lastSampleAt,
    required this.coverImagePath,
    required this.onTap,
    this.isUncollected = false,
    this.expanded = false,
  });

  /// Null only when [isUncollected] is true (the synthetic bucket).
  final Collection? collection;
  final bool isUncollected;
  final int sampleCount;
  final DateTime lastSampleAt;
  final String? coverImagePath;

  /// Controls the chevron rotation. The header is the same widget whether
  /// expanded or not — the section below it appears/disappears around it.
  final bool expanded;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final accent =
        isUncollected ? Colors.white54 : Colors.amberAccent;
    final name = isUncollected ? 'Uncollected' : collection!.name;
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
                      Icon(
                        isUncollected
                            ? Icons.inbox_outlined
                            : Icons.collections_bookmark_outlined,
                        size: 14,
                        color: accent,
                      ),
                      const SizedBox(width: 4),
                      Expanded(
                        child: Text(
                          name,
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
                    _subtitle(),
                    style: const TextStyle(
                        color: Colors.white54, fontSize: 13),
                  ),
                ],
              ),
            ),
            AnimatedRotation(
              turns: expanded ? 0.25 : 0,
              duration: const Duration(milliseconds: 150),
              child: const Icon(Icons.chevron_right, color: Colors.white38),
            ),
          ],
        ),
      ),
    );
  }

  String _subtitle() {
    if (isUncollected) {
      return sampleCount == 0
          ? 'No loose samples'
          : 'Latest ${_formatDate(lastSampleAt)}';
    }
    if (sampleCount == 0) {
      return 'Empty · created ${_formatDate(collection!.createdAt)}';
    }
    return 'Latest ${_formatDate(lastSampleAt)}';
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
  });

  final Sample sample;
  final VoidCallback onTap;
  final VoidCallback onLongPress;

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
                    Row(
                      children: [
                        const Icon(
                          Icons.place_outlined,
                          size: 14,
                          color: Colors.white54,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          '${sample.latitude!.toStringAsFixed(4)}, ${sample.longitude!.toStringAsFixed(4)}',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 13,
                          ),
                        ),
                      ],
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
  _OverlayMode _overlayMode = _OverlayMode.disease;

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
        // Disease analysis may have mutated the spike masks (Stage 4 cleanup);
        // prefer the cleaned overlay it returns over the pre-cleanup one we
        // showed during the rebuild step above.
        if (analyzed.segmentationOverlayPng != null) {
          _segmentationOverlayPng = analyzed.segmentationOverlayPng;
        }
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

  void _openFullscreen({
    required bool inDiseaseMode,
    required bool inSegmentationMode,
  }) {
    final initial = inDiseaseMode
        ? _OverlayMode.disease
        : (inSegmentationMode ? _OverlayMode.segmentation : _OverlayMode.bbox);
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => _FullscreenSampleViewer(
          imagePath: _sample.filePath,
          instances: _instances,
          workingWidth: _workingW ?? 1,
          workingHeight: _workingH ?? 1,
          segmentationOverlayPng: _segmentationOverlayPng,
          diseaseOverlayPng: _diseaseOverlayPng,
          initialMode: initial,
          hasDisease: _hasDiseaseData,
          hasInstances: _hasInstances,
        ),
      ),
    );
  }

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
          if (hasWorking)
            IconButton(
              tooltip: 'Fullscreen',
              icon: const Icon(Icons.fullscreen, color: Colors.white),
              onPressed: () => _openFullscreen(
                inDiseaseMode: inDiseaseMode,
                inSegmentationMode: inSegmentationMode,
              ),
            ),
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

/// Full-screen sample image inspector with seg/disease overlay toggles. The
/// regular [SampleViewerScreen] packs the image alongside the instance strip,
/// collection panel, etc.; this view drops all of that chrome so the user can
/// pan and zoom into the raw photo. The overlay toggle in the bottom action
/// bar mirrors the in-screen toggle so users keep the same mental model.
class _FullscreenSampleViewer extends StatefulWidget {
  const _FullscreenSampleViewer({
    required this.imagePath,
    required this.instances,
    required this.workingWidth,
    required this.workingHeight,
    required this.segmentationOverlayPng,
    required this.diseaseOverlayPng,
    required this.initialMode,
    required this.hasDisease,
    required this.hasInstances,
  });

  final String imagePath;
  final List<SampleInstance> instances;
  final int workingWidth;
  final int workingHeight;
  final Uint8List? segmentationOverlayPng;
  final Uint8List? diseaseOverlayPng;
  final _OverlayMode initialMode;
  final bool hasDisease;
  final bool hasInstances;

  @override
  State<_FullscreenSampleViewer> createState() =>
      _FullscreenSampleViewerState();
}

class _FullscreenSampleViewerState extends State<_FullscreenSampleViewer> {
  late _OverlayMode _mode = widget.initialMode;
  bool _showOverlay = true;

  @override
  Widget build(BuildContext context) {
    final aspect = widget.workingWidth / widget.workingHeight;
    final inDisease = _mode == _OverlayMode.disease && widget.hasDisease;
    final inSegmentation =
        _mode == _OverlayMode.segmentation && widget.hasInstances;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: const Text(
          'Sample',
          style: TextStyle(color: Colors.white, fontSize: 15),
        ),
        actions: [
          IconButton(
            tooltip: _showOverlay ? 'Hide overlay' : 'Show overlay',
            icon: Icon(
              _showOverlay ? Icons.visibility : Icons.visibility_off,
              color: Colors.white,
            ),
            onPressed: () => setState(() => _showOverlay = !_showOverlay),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: InteractiveViewer(
                  minScale: 1,
                  maxScale: 8,
                  child: AspectRatio(
                    aspectRatio: aspect,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.file(
                          File(widget.imagePath),
                          gaplessPlayback: true,
                          fit: BoxFit.fill,
                          cacheWidth: 4096,
                          errorBuilder: (_, __, ___) => const Icon(
                            Icons.broken_image_outlined,
                            color: Colors.white54,
                            size: 64,
                          ),
                        ),
                        if (_showOverlay && widget.hasInstances)
                          _fullscreenOverlay(
                            inDisease: inDisease,
                            inSegmentation: inSegmentation,
                          ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            if (widget.hasInstances)
              Padding(
                padding: const EdgeInsets.symmetric(
                    horizontal: 16, vertical: 12),
                child: _OverlayModeToggle(
                  mode: _mode,
                  hasDisease: widget.hasDisease,
                  onChanged: (m) => setState(() {
                    _mode = m;
                    _showOverlay = true;
                  }),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _fullscreenOverlay({
    required bool inDisease,
    required bool inSegmentation,
  }) {
    if (inDisease && widget.diseaseOverlayPng != null) {
      return Image.memory(
        widget.diseaseOverlayPng!,
        gaplessPlayback: true,
        fit: BoxFit.fill,
      );
    }
    if (inSegmentation && widget.segmentationOverlayPng != null) {
      return Image.memory(
        widget.segmentationOverlayPng!,
        gaplessPlayback: true,
        fit: BoxFit.fill,
      );
    }
    return _BboxOverlay(
      instances: widget.instances,
      width: widget.workingWidth,
      height: widget.workingHeight,
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
                '· ${status.elapsed.inSeconds} s',
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
