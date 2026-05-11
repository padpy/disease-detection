import 'package:flutter/material.dart';
import 'package:gopher_eye/model/collection.dart';
import 'package:gopher_eye/services/sample_repository.dart';

/// Result returned by [CollectionPickerScreen]. Wraps the picked collection
/// (which may be null to mean "None") together with an explicit `cleared`
/// flag so callers can distinguish "user dismissed without changing" (null
/// result from `Navigator.pop`) from "user explicitly chose None" (this
/// object with `cleared: true`).
class CollectionPickResult {
  const CollectionPickResult.collection(this.collection) : cleared = false;
  const CollectionPickResult.none()
      : collection = null,
        cleared = true;

  final Collection? collection;
  final bool cleared;
}

/// Modal screen for picking the "active" collection. Lists all existing
/// collections (newest first), supports text search, exposes a "None"
/// option, and lets the user create a new one inline. Used from the camera
/// screen to start a session and from the sample inspector to reassign an
/// existing capture.
class CollectionPickerScreen extends StatefulWidget {
  const CollectionPickerScreen({
    super.key,
    this.activeCollectionId,
    this.title = 'Collection',
    this.allowNone = true,
  });

  /// Currently-selected collection id (rendered with a check mark). Null when
  /// nothing is active.
  final int? activeCollectionId;

  final String title;

  /// When false, hides the "None" entry. The inspector uses this to require
  /// a positive choice (the user can pop instead to cancel).
  final bool allowNone;

  @override
  State<CollectionPickerScreen> createState() => _CollectionPickerScreenState();
}

class _CollectionPickerScreenState extends State<CollectionPickerScreen> {
  final _searchCtrl = TextEditingController();
  List<Collection> _collections = const [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _searchCtrl.addListener(_runSearch);
    _runSearch();
  }

  @override
  void dispose() {
    _searchCtrl.removeListener(_runSearch);
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _runSearch() async {
    final query = _searchCtrl.text;
    setState(() => _loading = true);
    try {
      final list =
          await SampleRepository.instance.searchCollections(query);
      if (!mounted) return;
      setState(() {
        _collections = list;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  Future<void> _createNew() async {
    final created = await showModalBottomSheet<Collection>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.grey[900],
      builder: (_) => const _NewCollectionSheet(),
    );
    if (created == null || !mounted) return;
    Navigator.of(context).pop(CollectionPickResult.collection(created));
  }

  void _pickNone() =>
      Navigator.of(context).pop(const CollectionPickResult.none());

  void _pick(Collection c) =>
      Navigator.of(context).pop(CollectionPickResult.collection(c));

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Text(widget.title,
            style: const TextStyle(color: Colors.white, fontSize: 16)),
        actions: [
          IconButton(
            tooltip: 'New collection',
            icon: const Icon(Icons.add, color: Colors.white),
            onPressed: _createNew,
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
              child: TextField(
                controller: _searchCtrl,
                style: const TextStyle(color: Colors.white),
                decoration: InputDecoration(
                  hintText: 'Search collections',
                  hintStyle: const TextStyle(color: Colors.white38),
                  prefixIcon:
                      const Icon(Icons.search, color: Colors.white54),
                  suffixIcon: _searchCtrl.text.isEmpty
                      ? null
                      : IconButton(
                          icon: const Icon(Icons.close,
                              color: Colors.white54),
                          onPressed: () => _searchCtrl.clear(),
                        ),
                  filled: true,
                  fillColor: Colors.white10,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                    borderSide: BorderSide.none,
                  ),
                ),
              ),
            ),
            Expanded(
              child: _loading
                  ? const Center(
                      child:
                          CircularProgressIndicator(color: Colors.white),
                    )
                  : ListView(
                      children: [
                        if (widget.allowNone)
                          ListTile(
                            leading: Icon(
                              widget.activeCollectionId == null
                                  ? Icons.radio_button_checked
                                  : Icons.radio_button_unchecked,
                              color: Colors.white,
                            ),
                            title: const Text(
                              'None',
                              style: TextStyle(color: Colors.white),
                            ),
                            subtitle: const Text(
                              'Don\'t group these captures',
                              style: TextStyle(
                                  color: Colors.white54, fontSize: 12),
                            ),
                            onTap: _pickNone,
                          ),
                        ListTile(
                          leading: const Icon(Icons.add_circle_outline,
                              color: Colors.lightBlueAccent),
                          title: const Text(
                            'New collection…',
                            style: TextStyle(
                                color: Colors.lightBlueAccent),
                          ),
                          onTap: _createNew,
                        ),
                        if (_collections.isEmpty &&
                            _searchCtrl.text.isNotEmpty)
                          const Padding(
                            padding: EdgeInsets.all(24),
                            child: Center(
                              child: Text(
                                'No matching collections',
                                style: TextStyle(color: Colors.white54),
                              ),
                            ),
                          ),
                        for (final c in _collections)
                          ListTile(
                            leading: Icon(
                              c.id == widget.activeCollectionId
                                  ? Icons.radio_button_checked
                                  : Icons.radio_button_unchecked,
                              color: Colors.white,
                            ),
                            title: Text(
                              c.name,
                              style: const TextStyle(color: Colors.white),
                            ),
                            subtitle: Text(
                              _formatDate(c.createdAt),
                              style: const TextStyle(
                                  color: Colors.white54, fontSize: 12),
                            ),
                            onTap: () => _pick(c),
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

/// Bottom sheet for creating a new collection. Pre-fills the name field with
/// the current date so the common "field visit on YYYY-MM-DD" workflow is a
/// single tap, but the user can edit it freely before saving.
class _NewCollectionSheet extends StatefulWidget {
  const _NewCollectionSheet();

  @override
  State<_NewCollectionSheet> createState() => _NewCollectionSheetState();
}

class _NewCollectionSheetState extends State<_NewCollectionSheet> {
  late final TextEditingController _nameCtrl =
      TextEditingController(text: _defaultName(DateTime.now()));
  bool _busy = false;

  @override
  void dispose() {
    _nameCtrl.dispose();
    super.dispose();
  }

  static String _defaultName(DateTime dt) {
    final m = dt.month.toString().padLeft(2, '0');
    final d = dt.day.toString().padLeft(2, '0');
    return '${dt.year}-$m-$d';
  }

  Future<void> _save() async {
    final name = _nameCtrl.text.trim();
    if (name.isEmpty || _busy) return;
    setState(() => _busy = true);
    try {
      final created =
          await SampleRepository.instance.createCollection(name: name);
      if (!mounted) return;
      Navigator.of(context).pop(created);
    } catch (e) {
      if (!mounted) return;
      setState(() => _busy = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not create collection: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).viewInsets.bottom;
    return Padding(
      padding: EdgeInsets.fromLTRB(16, 16, 16, 16 + bottomInset),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'New collection',
            style: TextStyle(color: Colors.white, fontSize: 16),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _nameCtrl,
            autofocus: true,
            style: const TextStyle(color: Colors.white),
            textInputAction: TextInputAction.done,
            onSubmitted: (_) => _save(),
            decoration: InputDecoration(
              labelText: 'Name',
              labelStyle: const TextStyle(color: Colors.white54),
              filled: true,
              fillColor: Colors.white10,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: BorderSide.none,
              ),
            ),
          ),
          const SizedBox(height: 16),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: Colors.white,
              foregroundColor: Colors.black,
            ),
            onPressed: _busy ? null : _save,
            child: _busy
                ? const SizedBox(
                    height: 18,
                    width: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.black,
                    ),
                  )
                : const Text('Create'),
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
  return '${months[dt.month - 1]} ${dt.day}, ${dt.year}';
}
