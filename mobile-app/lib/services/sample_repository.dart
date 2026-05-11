import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:geolocator/geolocator.dart';
import 'package:gopher_eye/model/chat_turn.dart';
import 'package:gopher_eye/model/collection.dart';
import 'package:gopher_eye/model/detection_mode.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/model/sample_instance.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

class SampleRepository {
  SampleRepository._();
  static final SampleRepository instance = SampleRepository._();

  static const _dbName = 'gopher_eye.db';
  static const _table = 'samples';
  static const _instanceTable = 'sample_instances';
  static const _chatTable = 'chat_messages';
  static const _collectionTable = 'collections';
  static const _dbVersion = 10;

  /// Columns consumed by `Sample.fromMap`. Used as the explicit projection
  /// for queries that hydrate `Sample` objects so we don't pull the
  /// (multi-MB) `working_image_png`, `disease_overlay_png`, and
  /// `segmentation_overlay_png` blobs over the platform channel for every
  /// row in the samples list.
  static const List<String> _sampleDisplayColumns = [
    'id',
    'file_path',
    'taken_at',
    'latitude',
    'longitude',
    'accuracy',
    'detection_mode',
    'collection_id',
    'qr_id',
    'qr_line',
    'qr_rep',
    'qr_location',
    'qr_note',
  ];

  Database? _db;
  Directory? _samplesDir;

  Future<Database> _open() async {
    if (_db != null) return _db!;
    final dbPath = p.join(await getDatabasesPath(), _dbName);
    final db = await openDatabase(
      dbPath,
      version: _dbVersion,
      onCreate: (db, version) async {
        await _createCollectionsTable(db);
        await _createSamplesTable(db);
        await _createInstancesTable(db);
        await _createChatTable(db);
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN working_image_w INTEGER',
          );
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN working_image_h INTEGER',
          );
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN working_image_png BLOB',
          );
          await _createInstancesTable(db);
        }
        if (oldVersion < 4) {
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN segmentation_overlay_png BLOB',
          );
        }
        if (oldVersion < 5) {
          await db.execute(
            "ALTER TABLE $_table ADD COLUMN detection_mode TEXT NOT NULL DEFAULT '${DetectionMode.wheatFhb.id}'",
          );
        }
        if (oldVersion < 6) {
          await _createChatTable(db);
        }
        if (oldVersion < 7) {
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN remote_id INTEGER',
          );
          await db.execute(
            'ALTER TABLE $_instanceTable ADD COLUMN remote_id INTEGER',
          );
        }
        if (oldVersion < 8) {
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN qr_name TEXT',
          );
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN qr_id TEXT',
          );
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN qr_note TEXT',
          );
        }
        if (oldVersion < 9) {
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN qr_line TEXT',
          );
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN qr_rep TEXT',
          );
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN qr_location TEXT',
          );
        }
        if (oldVersion < 10) {
          await _createCollectionsTable(db);
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN collection_id INTEGER',
          );
          await db.execute(
            'CREATE INDEX IF NOT EXISTS idx_samples_collection '
            'ON $_table(collection_id)',
          );
        }
        if (oldVersion < 3) {
          await db.execute(
            'ALTER TABLE $_table ADD COLUMN disease_overlay_png BLOB',
          );
          await db.execute(
            'ALTER TABLE $_instanceTable ADD COLUMN fhb_green INTEGER',
          );
          await db.execute(
            'ALTER TABLE $_instanceTable ADD COLUMN fhb_necrotic INTEGER',
          );
          await db.execute(
            'ALTER TABLE $_instanceTable ADD COLUMN fhb_other INTEGER',
          );
          await db.execute(
            'ALTER TABLE $_instanceTable ADD COLUMN fhb_total INTEGER',
          );
          await db.execute(
            'ALTER TABLE $_instanceTable ADD COLUMN fhb_ratio REAL',
          );
          await db.execute(
            'ALTER TABLE $_instanceTable ADD COLUMN fhb_severity TEXT',
          );
          await db.execute(
            'ALTER TABLE $_instanceTable ADD COLUMN disease_preview_png BLOB',
          );
        }
      },
    );
    _db = db;
    return _db!;
  }

  Future<void> _createSamplesTable(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS $_table (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        taken_at INTEGER NOT NULL,
        latitude REAL,
        longitude REAL,
        accuracy REAL,
        working_image_w INTEGER,
        working_image_h INTEGER,
        working_image_png BLOB,
        disease_overlay_png BLOB,
        segmentation_overlay_png BLOB,
        detection_mode TEXT NOT NULL DEFAULT '${DetectionMode.wheatFhb.id}',
        remote_id INTEGER,
        collection_id INTEGER,
        qr_id TEXT,
        qr_line TEXT,
        qr_rep TEXT,
        qr_location TEXT,
        qr_note TEXT
      )
    ''');
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_samples_taken_at ON $_table(taken_at DESC)',
    );
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_samples_collection '
      'ON $_table(collection_id)',
    );
  }

  Future<void> _createCollectionsTable(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS $_collectionTable (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at INTEGER NOT NULL
      )
    ''');
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_collections_created_at '
      'ON $_collectionTable(created_at DESC)',
    );
  }

  Future<void> _createInstancesTable(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS $_instanceTable (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id INTEGER NOT NULL,
        idx INTEGER NOT NULL,
        bbox_left REAL NOT NULL,
        bbox_top REAL NOT NULL,
        bbox_right REAL NOT NULL,
        bbox_bottom REAL NOT NULL,
        centroid_x REAL NOT NULL,
        centroid_y REAL NOT NULL,
        score REAL NOT NULL,
        image_w INTEGER NOT NULL,
        image_h INTEGER NOT NULL,
        mask_png BLOB NOT NULL,
        preview_png BLOB NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        fhb_green INTEGER,
        fhb_necrotic INTEGER,
        fhb_other INTEGER,
        fhb_total INTEGER,
        fhb_ratio REAL,
        fhb_severity TEXT,
        disease_preview_png BLOB,
        remote_id INTEGER
      )
    ''');
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_instances_sample ON $_instanceTable(sample_id, idx)',
    );
  }

  Future<void> _createChatTable(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS $_chatTable (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at INTEGER NOT NULL
      )
    ''');
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_chat_instance ON $_chatTable(instance_id, created_at ASC)',
    );
  }

  Future<Directory> _samplesDirectory() async {
    if (_samplesDir != null) return _samplesDir!;
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(docs.path, 'samples'));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    _samplesDir = dir;
    return dir;
  }

  Future<Sample> saveCapture({
    required XFile image,
    Position? position,
    DetectionMode detectionMode = DetectionMode.wheatFhb,
    int? collectionId,
    String? qrId,
    String? qrLine,
    String? qrRep,
    String? qrLocation,
    String? qrNote,
  }) async {
    final dir = await _samplesDirectory();
    final filename = '${DateTime.now().microsecondsSinceEpoch}.jpg';
    final targetPath = p.join(dir.path, filename);
    await image.saveTo(targetPath);

    final sample = Sample(
      filePath: targetPath,
      takenAt: DateTime.now(),
      latitude: position?.latitude,
      longitude: position?.longitude,
      accuracy: position?.accuracy,
      detectionMode: detectionMode,
      collectionId: collectionId,
      qrId: qrId,
      qrLine: qrLine,
      qrRep: qrRep,
      qrLocation: qrLocation,
      qrNote: qrNote,
    );

    try {
      final db = await _open();
      final id = await db.insert(_table, sample.toMap());
      return sample.copyWith(id: id);
    } catch (e) {
      // Orphan file cleanup if DB insert fails.
      try {
        await File(targetPath).delete();
      } catch (_) {}
      rethrow;
    }
  }

  Future<List<Sample>> listAll() async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: _sampleDisplayColumns,
      orderBy: 'taken_at DESC',
    );
    return rows.map(Sample.fromMap).toList();
  }

  Future<Sample?> findById(int id) async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: _sampleDisplayColumns,
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    return Sample.fromMap(rows.first);
  }

  /// Persist a new [DetectionMode] for [sampleId] and return the resulting
  /// row. Used when the user switches modes from the sample viewer.
  Future<Sample?> updateDetectionMode({
    required int sampleId,
    required DetectionMode mode,
  }) async {
    final db = await _open();
    await db.update(
      _table,
      {'detection_mode': mode.id},
      where: 'id = ?',
      whereArgs: [sampleId],
    );
    return findById(sampleId);
  }

  /// Overwrite the QR-derived metadata (id, line, rep, location, note) for
  /// [sampleId]. Pass `null` for any field to clear it. Returns the updated
  /// row. Used by the inspector's "Sample tag" editor.
  Future<Sample?> updateQrMetadata({
    required int sampleId,
    required String? qrId,
    required String? qrLine,
    required String? qrRep,
    required String? qrLocation,
    required String? qrNote,
  }) async {
    final db = await _open();
    await db.update(
      _table,
      {
        'qr_id': qrId,
        'qr_line': qrLine,
        'qr_rep': qrRep,
        'qr_location': qrLocation,
        'qr_note': qrNote,
      },
      where: 'id = ?',
      whereArgs: [sampleId],
    );
    return findById(sampleId);
  }

  Future<void> delete(int id) async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: ['file_path'],
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );
    final instanceRows = await db.query(
      _instanceTable,
      columns: ['id'],
      where: 'sample_id = ?',
      whereArgs: [id],
    );
    for (final row in instanceRows) {
      final instanceId = row['id'] as int?;
      if (instanceId != null) {
        await db.delete(
          _chatTable,
          where: 'instance_id = ?',
          whereArgs: [instanceId],
        );
      }
    }
    await db.delete(_instanceTable, where: 'sample_id = ?', whereArgs: [id]);
    await db.delete(_table, where: 'id = ?', whereArgs: [id]);
    if (rows.isNotEmpty) {
      final path = rows.first['file_path'] as String?;
      if (path != null) {
        try {
          await File(path).delete();
        } catch (_) {
          // File may already be gone; ignore.
        }
      }
    }
  }

  // ---------- Working image (cached SAM-resolution PNG) ----------

  /// Persist the SAM-resolution working image for [sampleId] so the editor
  /// can re-run the encoder without re-decoding the original capture.
  Future<void> saveWorkingImage({
    required int sampleId,
    required Uint8List png,
    required int width,
    required int height,
  }) async {
    final db = await _open();
    await db.update(
      _table,
      {
        'working_image_png': png,
        'working_image_w': width,
        'working_image_h': height,
      },
      where: 'id = ?',
      whereArgs: [sampleId],
    );
  }

  Future<({Uint8List png, int width, int height})?> loadWorkingImage(
    int sampleId,
  ) async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: ['working_image_png', 'working_image_w', 'working_image_h'],
      where: 'id = ?',
      whereArgs: [sampleId],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    final png = rows.first['working_image_png'] as Uint8List?;
    final w = rows.first['working_image_w'] as int?;
    final h = rows.first['working_image_h'] as int?;
    if (png == null || w == null || h == null) return null;
    return (png: png, width: w, height: h);
  }

  /// Persist the combined per-pixel disease overlay (green/red/yellow tints
  /// across every instance) so the viewer can paint disease-mode without
  /// re-running classification on hydration.
  Future<void> saveDiseaseOverlay({
    required int sampleId,
    required Uint8List? png,
  }) async {
    final db = await _open();
    await db.update(
      _table,
      {'disease_overlay_png': png},
      where: 'id = ?',
      whereArgs: [sampleId],
    );
  }

  Future<Uint8List?> loadDiseaseOverlay(int sampleId) async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: ['disease_overlay_png'],
      where: 'id = ?',
      whereArgs: [sampleId],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    return rows.first['disease_overlay_png'] as Uint8List?;
  }

  /// Persist the combined segmentation overlay (all instance masks tinted
  /// green and outlined). Drawn over the displayed image when the user
  /// selects "Segmentation" mode in the viewer.
  Future<void> saveSegmentationOverlay({
    required int sampleId,
    required Uint8List? png,
  }) async {
    final db = await _open();
    await db.update(
      _table,
      {'segmentation_overlay_png': png},
      where: 'id = ?',
      whereArgs: [sampleId],
    );
  }

  Future<Uint8List?> loadSegmentationOverlay(int sampleId) async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: ['segmentation_overlay_png'],
      where: 'id = ?',
      whereArgs: [sampleId],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    return rows.first['segmentation_overlay_png'] as Uint8List?;
  }

  // ---------- Sample instances ----------

  Future<List<SampleInstance>> listInstances(int sampleId) async {
    final db = await _open();
    final rows = await db.query(
      _instanceTable,
      where: 'sample_id = ?',
      whereArgs: [sampleId],
      orderBy: 'idx ASC',
    );
    return rows.map(SampleInstance.fromMap).toList();
  }

  /// Replaces the persisted instances for [sampleId] with [instances].
  /// Atomic — either every instance lands or none do.
  Future<List<SampleInstance>> replaceInstances({
    required int sampleId,
    required List<SampleInstance> instances,
  }) async {
    final db = await _open();
    final saved = <SampleInstance>[];
    await db.transaction((txn) async {
      await txn.delete(
        _instanceTable,
        where: 'sample_id = ?',
        whereArgs: [sampleId],
      );
      for (final inst in instances) {
        final id = await txn.insert(
          _instanceTable,
          inst.toMap()..remove('id'),
        );
        saved.add(inst.copyWith(id: id));
      }
    });
    return saved;
  }

  /// Inserts [instance] at the next available `idx` for its sample. Use this
  /// for user-created instances that don't come from a detection sweep.
  Future<SampleInstance> createInstance(SampleInstance instance) async {
    final db = await _open();
    return db.transaction((txn) async {
      final res = await txn.rawQuery(
        'SELECT COALESCE(MAX(idx), -1) + 1 AS next FROM $_instanceTable WHERE sample_id = ?',
        [instance.sampleId],
      );
      final nextIdx = (res.first['next'] as int?) ?? 0;
      final now = DateTime.now();
      final toInsert = SampleInstance(
        sampleId: instance.sampleId,
        idx: nextIdx,
        bbox: instance.bbox,
        centroid: instance.centroid,
        score: instance.score,
        imageWidth: instance.imageWidth,
        imageHeight: instance.imageHeight,
        maskPng: instance.maskPng,
        previewPng: instance.previewPng,
        createdAt: now,
        updatedAt: now,
      );
      final id = await txn.insert(_instanceTable, toInsert.toMap()..remove('id'));
      return toInsert.copyWith(id: id);
    });
  }

  Future<SampleInstance> updateInstance(SampleInstance instance) async {
    if (instance.id == null) {
      throw ArgumentError('updateInstance requires an instance id');
    }
    final db = await _open();
    final updated = instance.copyWith(updatedAt: DateTime.now());
    await db.update(
      _instanceTable,
      updated.toMap()..remove('id'),
      where: 'id = ?',
      whereArgs: [instance.id],
    );
    return updated;
  }

  Future<void> deleteInstance(int id) async {
    final db = await _open();
    await db.delete(_chatTable, where: 'instance_id = ?', whereArgs: [id]);
    await db.delete(_instanceTable, where: 'id = ?', whereArgs: [id]);
  }

  // ---------- Chat history (per instance) ----------

  /// Returns the chat transcript for [instanceId] in chronological order.
  Future<List<ChatTurn>> listChatTurns(int instanceId) async {
    final db = await _open();
    final rows = await db.query(
      _chatTable,
      where: 'instance_id = ?',
      whereArgs: [instanceId],
      orderBy: 'created_at ASC, id ASC',
    );
    return rows.map(ChatTurn.fromMap).toList();
  }

  /// Persists [turn] and returns it with the assigned row id.
  Future<ChatTurn> appendChatTurn(ChatTurn turn) async {
    final db = await _open();
    final id = await db.insert(_chatTable, turn.toMap()..remove('id'));
    return turn.copyWith(id: id);
  }

  /// Removes every turn for [instanceId]. Used by the "Clear chat" action.
  Future<void> clearChat(int instanceId) async {
    final db = await _open();
    await db.delete(
      _chatTable,
      where: 'instance_id = ?',
      whereArgs: [instanceId],
    );
  }

  /// Drops every turn for [instanceId] whose row id is strictly greater than
  /// [turnId]. Used by the "Resend" action to rewind history before re-asking
  /// the LLM, so the prior user prompt and everything before it stays put
  /// while any subsequent (now-stale) replies disappear.
  Future<void> deleteChatTurnsAfterId({
    required int instanceId,
    required int turnId,
  }) async {
    final db = await _open();
    await db.delete(
      _chatTable,
      where: 'instance_id = ? AND id > ?',
      whereArgs: [instanceId, turnId],
    );
  }

  /// Returns instance ids that have at least one chat turn. Lets the picker
  /// surface "instances with conversations" without N round-trips.
  Future<Set<int>> instancesWithChat() async {
    final db = await _open();
    final rows = await db.rawQuery(
      'SELECT DISTINCT instance_id FROM $_chatTable',
    );
    return rows
        .map((r) => r['instance_id'] as int)
        .toSet();
  }

  // ---------- Server sync ----------

  /// Persist the server-side row id for [sampleId] so future updates target
  /// the existing remote record instead of re-creating it.
  Future<void> setSampleRemoteId(int sampleId, int? remoteId) async {
    final db = await _open();
    await db.update(
      _table,
      {'remote_id': remoteId},
      where: 'id = ?',
      whereArgs: [sampleId],
    );
  }

  Future<int?> sampleRemoteId(int sampleId) async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: ['remote_id'],
      where: 'id = ?',
      whereArgs: [sampleId],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    return rows.first['remote_id'] as int?;
  }

  Future<int?> findSampleByRemoteId(int remoteId) async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: ['id'],
      where: 'remote_id = ?',
      whereArgs: [remoteId],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    return rows.first['id'] as int?;
  }

  /// Local samples that haven't been pushed to a server yet.
  Future<List<Sample>> listUnsynced() async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: _sampleDisplayColumns,
      where: 'remote_id IS NULL',
      orderBy: 'taken_at DESC',
    );
    return rows.map(Sample.fromMap).toList();
  }

  // ---------- Collections ----------

  /// Returns every collection, newest first.
  Future<List<Collection>> listCollections() async {
    final db = await _open();
    final rows = await db.query(
      _collectionTable,
      orderBy: 'created_at DESC',
    );
    return rows.map(Collection.fromMap).toList();
  }

  /// Case-insensitive substring match against collection names. An empty
  /// [query] is treated as "list all" so callers can use this for both
  /// browse and search without branching.
  Future<List<Collection>> searchCollections(String query) async {
    final trimmed = query.trim();
    if (trimmed.isEmpty) return listCollections();
    final db = await _open();
    final rows = await db.query(
      _collectionTable,
      where: 'LOWER(name) LIKE ?',
      whereArgs: ['%${trimmed.toLowerCase()}%'],
      orderBy: 'created_at DESC',
    );
    return rows.map(Collection.fromMap).toList();
  }

  Future<Collection?> findCollection(int id) async {
    final db = await _open();
    final rows = await db.query(
      _collectionTable,
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    return Collection.fromMap(rows.first);
  }

  Future<Collection> createCollection({
    required String name,
    DateTime? createdAt,
  }) async {
    final db = await _open();
    final collection = Collection(
      name: name,
      createdAt: createdAt ?? DateTime.now(),
    );
    final id = await db.insert(_collectionTable, collection.toMap());
    return collection.copyWith(id: id);
  }

  /// Reassign (or clear, via null) the collection of a sample. Returns the
  /// updated row.
  Future<Sample?> setSampleCollection({
    required int sampleId,
    required int? collectionId,
  }) async {
    final db = await _open();
    await db.update(
      _table,
      {'collection_id': collectionId},
      where: 'id = ?',
      whereArgs: [sampleId],
    );
    return findById(sampleId);
  }

  /// Sample count grouped by collection id. Includes a `null` key for the
  /// "Uncollected" bucket so the samples screen can show one row per
  /// collection plus the loose samples without N+1 queries.
  Future<Map<int?, int>> sampleCountsByCollection() async {
    final db = await _open();
    final rows = await db.rawQuery(
      'SELECT collection_id, COUNT(*) AS n FROM $_table GROUP BY collection_id',
    );
    final out = <int?, int>{};
    for (final row in rows) {
      out[row['collection_id'] as int?] = (row['n'] as int?) ?? 0;
    }
    return out;
  }

  /// Samples belonging to [collectionId]. Pass `null` to fetch the
  /// "Uncollected" bucket.
  Future<List<Sample>> listSamplesInCollection(int? collectionId) async {
    final db = await _open();
    final rows = await db.query(
      _table,
      columns: _sampleDisplayColumns,
      where: collectionId == null
          ? 'collection_id IS NULL'
          : 'collection_id = ?',
      whereArgs: collectionId == null ? null : [collectionId],
      orderBy: 'taken_at DESC',
    );
    return rows.map(Sample.fromMap).toList();
  }

  /// Insert a sample whose row id is set explicitly (used when hydrating from
  /// a server pull so the local id mirrors the server's id, simplifying
  /// downstream lookups). Returns the inserted row id; throws on conflict.
  Future<int> insertSampleFromRemote({
    required int remoteId,
    required String filePath,
    required DateTime takenAt,
    required DetectionMode detectionMode,
    double? latitude,
    double? longitude,
    double? accuracy,
  }) async {
    final db = await _open();
    return db.insert(_table, {
      'file_path': filePath,
      'taken_at': takenAt.millisecondsSinceEpoch,
      'latitude': latitude,
      'longitude': longitude,
      'accuracy': accuracy,
      'detection_mode': detectionMode.id,
      'remote_id': remoteId,
    });
  }
}
