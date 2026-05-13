import 'package:flutter/material.dart';
import 'package:gopher_eye/model/collection.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/screens/samples_screen.dart';
import 'package:gopher_eye/services/detection_service.dart';
import 'package:gopher_eye/services/sample_repository.dart';

/// Drill-in screen for a single [Collection]. Reuses [SampleTile] +
/// [SampleViewerScreen] from the main samples screen so the in-collection
/// experience matches the ungrouped one — the only difference is the list
/// is filtered to the collection's samples.
class CollectionSamplesScreen extends StatefulWidget {
  const CollectionSamplesScreen({super.key, required this.collection});

  final Collection collection;

  @override
  State<CollectionSamplesScreen> createState() =>
      _CollectionSamplesScreenState();
}

class _CollectionSamplesScreenState extends State<CollectionSamplesScreen> {
  late Future<List<Sample>> _future;
  List<Sample> _samples = [];

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
    if (mounted) setState(() {});
  }

  Future<List<Sample>> _load() async {
    try {
      final list = await SampleRepository.instance
          .listSamplesInCollection(widget.collection.id);
      if (mounted) setState(() => _samples = list);
      return list;
    } catch (e, st) {
      debugPrint('[collection-samples] load failed: $e\n$st');
      rethrow;
    }
  }

  void _refresh() {
    setState(() {
      _samples = [];
      _future = _load();
    });
  }

  Future<void> _delete(Sample sample) async {
    if (sample.id == null) return;
    await SampleRepository.instance.delete(sample.id!);
    DetectionService.instance.forget(sample.id!);
    if (!mounted) return;
    setState(() => _samples.removeWhere((s) => s.id == sample.id));
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
              leading:
                  const Icon(Icons.delete_outline, color: Colors.redAccent),
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
    // Sample's collection might have changed inside the viewer; refresh
    // in case it no longer belongs here.
    if (mounted) _refresh();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Text(
          widget.collection.name,
          style: const TextStyle(color: Colors.white, fontSize: 16),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: _refresh,
          ),
        ],
      ),
      body: FutureBuilder<List<Sample>>(
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
          if (_samples.isEmpty) {
            return const Center(
              child: Text(
                'No samples in this collection yet',
                style: TextStyle(color: Colors.white70),
              ),
            );
          }
          return ListView.separated(
            itemCount: _samples.length,
            separatorBuilder: (_, __) =>
                const Divider(height: 1, color: Colors.white12),
            itemBuilder: (context, index) {
              final sample = _samples[index];
              return Dismissible(
                key: ValueKey('coll-sample-${sample.id}'),
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
                ),
              );
            },
          );
        },
      ),
    );
  }
}
