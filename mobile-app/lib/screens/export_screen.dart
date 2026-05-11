import 'dart:io';

import 'package:flutter/material.dart';
import 'package:gopher_eye/model/collection.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/services/export_destination.dart';
import 'package:gopher_eye/services/export_service.dart';
import 'package:gopher_eye/services/sample_repository.dart';

/// Multi-select picker for building a CSV (and optional COCO) export. The user
/// ticks any combination of whole collections and individual uncollected
/// samples, opts in to image+mask packaging, then picks a destination.
class ExportScreen extends StatefulWidget {
  const ExportScreen({super.key});

  @override
  State<ExportScreen> createState() => _ExportScreenState();
}

class _ExportScreenState extends State<ExportScreen> {
  late Future<void> _loadFuture;
  List<Collection> _collections = [];
  Map<int, List<Sample>> _samplesByCollection = const {};
  List<Sample> _uncollected = const [];

  final Set<int> _selectedCollectionIds = {};
  final Set<int> _selectedSampleIds = {};
  bool _includeImagesAndMasks = false;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _loadFuture = _load();
  }

  Future<void> _load() async {
    final repo = SampleRepository.instance;
    final collections = await repo.listCollections();
    final all = await repo.listAll();
    final byCollection = <int, List<Sample>>{};
    final uncollected = <Sample>[];
    for (final s in all) {
      if (s.id == null) continue;
      if (s.collectionId == null) {
        uncollected.add(s);
      } else {
        (byCollection[s.collectionId!] ??= []).add(s);
      }
    }
    if (!mounted) return;
    setState(() {
      _collections = collections;
      _samplesByCollection = byCollection;
      _uncollected = uncollected;
    });
  }

  int get _selectionCount {
    final fromCollections = _selectedCollectionIds.fold<int>(
      0,
      (acc, id) => acc + (_samplesByCollection[id]?.length ?? 0),
    );
    return fromCollections + _selectedSampleIds.length;
  }

  void _toggleCollection(Collection collection) {
    final id = collection.id;
    if (id == null) return;
    setState(() {
      if (_selectedCollectionIds.contains(id)) {
        _selectedCollectionIds.remove(id);
      } else {
        _selectedCollectionIds.add(id);
      }
    });
  }

  void _toggleSample(Sample sample) {
    final id = sample.id;
    if (id == null) return;
    setState(() {
      if (_selectedSampleIds.contains(id)) {
        _selectedSampleIds.remove(id);
      } else {
        _selectedSampleIds.add(id);
      }
    });
  }

  Future<ExportDestination?> _pickDestination() async {
    return showModalBottomSheet<ExportDestination>(
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
                  'Send export to',
                  style: TextStyle(color: Colors.white, fontSize: 14),
                ),
              ),
            ),
            for (final d in ExportDestination.values)
              ListTile(
                leading: Icon(d.icon, color: Colors.white),
                title: Text(
                  d.label,
                  style: const TextStyle(color: Colors.white),
                ),
                onTap: () => Navigator.of(ctx).pop(d),
              ),
            ListTile(
              leading: const Icon(Icons.close, color: Colors.white70),
              title: const Text('Cancel',
                  style: TextStyle(color: Colors.white70)),
              onTap: () => Navigator.of(ctx).pop(),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _run() async {
    if (_busy || _selectionCount == 0) return;
    final destination = await _pickDestination();
    if (destination == null || !mounted) return;

    setState(() => _busy = true);
    final messenger = ScaffoldMessenger.of(context);
    final origin = _shareOrigin(context);
    ExportArtifact? artifact;
    try {
      final samples = await ExportService.instance.resolveSelection(
        sampleIds: _selectedSampleIds,
        collectionIds: _selectedCollectionIds,
      );
      if (samples.isEmpty) {
        if (!mounted) return;
        messenger.showSnackBar(
          const SnackBar(content: Text('Selection contains no samples.')),
        );
        return;
      }
      artifact = _includeImagesAndMasks
          ? await ExportService.instance.buildCsvWithCocoBundle(samples)
          : await ExportService.instance.buildCsvOnly(samples);

      final delivered = await deliverExport(
        artifact: artifact,
        destination: destination,
        sharePositionOrigin: origin,
      );

      if (!mounted) return;
      messenger.showSnackBar(
        SnackBar(
          content: Text(
            delivered
                ? 'Export ready: ${artifact.suggestedName}'
                : 'Export cancelled',
          ),
        ),
      );
    } catch (e, st) {
      debugPrint('[export] failed: $e\n$st');
      if (!mounted) return;
      messenger.showSnackBar(
        SnackBar(content: Text('Export failed: $e')),
      );
    } finally {
      if (artifact != null) {
        await cleanupExport(artifact);
      }
      if (mounted) setState(() => _busy = false);
    }
  }

  /// iPad popovers for share sheets need an anchoring rect; we use the
  /// run-button's slot so the sheet appears next to the action that
  /// triggered it.
  Rect? _shareOrigin(BuildContext context) {
    final box = context.findRenderObject();
    if (box is! RenderBox) return null;
    final position = box.localToGlobal(Offset.zero);
    return position & box.size;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: const Text(
          'Export samples',
          style: TextStyle(color: Colors.white, fontSize: 16),
        ),
      ),
      body: FutureBuilder<void>(
        future: _loadFuture,
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
                  'Failed to load: ${snapshot.error}',
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.redAccent),
                ),
              ),
            );
          }

          return SafeArea(
            child: Column(
              children: [
                _OptionsCard(
                  includeImagesAndMasks: _includeImagesAndMasks,
                  onChanged: (v) =>
                      setState(() => _includeImagesAndMasks = v),
                ),
                Expanded(child: _buildList()),
                _Footer(
                  selectionCount: _selectionCount,
                  busy: _busy,
                  onRun: _run,
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildList() {
    if (_collections.isEmpty && _uncollected.isEmpty) {
      return const Center(
        child: Text(
          'No samples to export yet.',
          style: TextStyle(color: Colors.white70),
        ),
      );
    }
    return ListView(
      padding: const EdgeInsets.symmetric(vertical: 8),
      children: [
        if (_collections.isNotEmpty)
          const _SectionHeader(text: 'Collections'),
        for (final c in _collections)
          _CollectionRow(
            collection: c,
            sampleCount: _samplesByCollection[c.id]?.length ?? 0,
            checked: _selectedCollectionIds.contains(c.id),
            coverImagePath: _samplesByCollection[c.id]?.isNotEmpty == true
                ? _samplesByCollection[c.id]!.first.filePath
                : null,
            onTap: () => _toggleCollection(c),
          ),
        if (_uncollected.isNotEmpty)
          const _SectionHeader(text: 'Uncollected samples'),
        for (final s in _uncollected)
          _SampleRow(
            sample: s,
            checked: _selectedSampleIds.contains(s.id),
            onTap: () => _toggleSample(s),
          ),
      ],
    );
  }
}

class _OptionsCard extends StatelessWidget {
  const _OptionsCard({
    required this.includeImagesAndMasks,
    required this.onChanged,
  });

  final bool includeImagesAndMasks;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white24),
      ),
      child: Row(
        children: [
          const Icon(Icons.image_outlined, color: Colors.white70, size: 18),
          const SizedBox(width: 10),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Include images & masks',
                  style: TextStyle(color: Colors.white, fontSize: 13),
                ),
                SizedBox(height: 2),
                Text(
                  'Bundles original images plus a COCO segmentation dataset (.zip).',
                  style: TextStyle(color: Colors.white54, fontSize: 11),
                ),
              ],
            ),
          ),
          Switch(
            value: includeImagesAndMasks,
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

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 6),
      child: Text(
        text.toUpperCase(),
        style: const TextStyle(
          color: Colors.white54,
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.8,
        ),
      ),
    );
  }
}

class _CollectionRow extends StatelessWidget {
  const _CollectionRow({
    required this.collection,
    required this.sampleCount,
    required this.checked,
    required this.coverImagePath,
    required this.onTap,
  });

  final Collection collection;
  final int sampleCount;
  final bool checked;
  final String? coverImagePath;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            _Checkbox(checked: checked),
            const SizedBox(width: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: SizedBox(
                width: 44,
                height: 44,
                child: coverImagePath != null
                    ? Image.file(
                        File(coverImagePath!),
                        fit: BoxFit.cover,
                        cacheWidth: 132,
                        cacheHeight: 132,
                        errorBuilder: (_, __, ___) =>
                            const _CollectionPlaceholder(),
                      )
                    : const _CollectionPlaceholder(),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    collection.name,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    '$sampleCount '
                    '${sampleCount == 1 ? 'sample' : 'samples'}',
                    style: const TextStyle(
                        color: Colors.white54, fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SampleRow extends StatelessWidget {
  const _SampleRow({
    required this.sample,
    required this.checked,
    required this.onTap,
  });

  final Sample sample;
  final bool checked;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          children: [
            _Checkbox(checked: checked),
            const SizedBox(width: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.file(
                File(sample.filePath),
                width: 44,
                height: 44,
                fit: BoxFit.cover,
                cacheWidth: 132,
                cacheHeight: 132,
                errorBuilder: (_, __, ___) => Container(
                  width: 44,
                  height: 44,
                  color: Colors.grey[800],
                  child: const Icon(
                    Icons.broken_image_outlined,
                    color: Colors.white54,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _shortDate(sample.takenAt),
                    style: const TextStyle(
                        color: Colors.white, fontSize: 13),
                  ),
                  Text(
                    sample.detectionMode.label +
                        (sample.hasQrMetadata ? ' · tagged' : ''),
                    style: const TextStyle(
                        color: Colors.white54, fontSize: 11),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Checkbox extends StatelessWidget {
  const _Checkbox({required this.checked});
  final bool checked;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 22,
      height: 22,
      decoration: BoxDecoration(
        color: checked ? Colors.white : Colors.transparent,
        borderRadius: BorderRadius.circular(4),
        border: Border.all(
          color: checked ? Colors.white : Colors.white54,
          width: 1.4,
        ),
      ),
      child: checked
          ? const Icon(Icons.check, size: 16, color: Colors.black)
          : null,
    );
  }
}

class _CollectionPlaceholder extends StatelessWidget {
  const _CollectionPlaceholder();

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white10,
      alignment: Alignment.center,
      child: const Icon(
        Icons.collections_bookmark_outlined,
        color: Colors.white38,
        size: 18,
      ),
    );
  }
}

class _Footer extends StatelessWidget {
  const _Footer({
    required this.selectionCount,
    required this.busy,
    required this.onRun,
  });

  final int selectionCount;
  final bool busy;
  final VoidCallback onRun;

  @override
  Widget build(BuildContext context) {
    final hasSelection = selectionCount > 0;
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: Colors.white12)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              hasSelection
                  ? '$selectionCount '
                      '${selectionCount == 1 ? 'sample' : 'samples'} selected'
                  : 'Pick at least one collection or sample',
              style: const TextStyle(color: Colors.white70, fontSize: 13),
            ),
          ),
          FilledButton.icon(
            style: FilledButton.styleFrom(
              backgroundColor: Colors.white,
              foregroundColor: Colors.black,
            ),
            onPressed: hasSelection && !busy ? onRun : null,
            icon: busy
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.black,
                    ),
                  )
                : const Icon(Icons.ios_share),
            label: Text(busy ? 'Building…' : 'Export'),
          ),
        ],
      ),
    );
  }
}

String _shortDate(DateTime dt) {
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  return '${months[dt.month - 1]} ${dt.day}, ${dt.year}';
}
