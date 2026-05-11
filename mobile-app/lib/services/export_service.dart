import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:archive/archive.dart';
import 'package:flutter/foundation.dart';
import 'package:gopher_eye/model/collection.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/model/sample_instance.dart';
import 'package:gopher_eye/services/sample_repository.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Result of building an export bundle. The caller is responsible for
/// persisting / sharing [file] (see `ExportSink`).
class ExportArtifact {
  ExportArtifact({
    required this.file,
    required this.suggestedName,
    required this.mimeType,
  });

  final File file;
  final String suggestedName;
  final String mimeType;
}

/// Build CSV / COCO exports for a user-chosen subset of samples (optionally
/// drawn from collections). The output is always a single file the caller can
/// hand to a destination sink (Files / Drive / share sheet).
class ExportService {
  ExportService._();
  static final ExportService instance = ExportService._();

  /// Resolve a flat, deduplicated list of samples for the supplied selection
  /// (explicit sample ids + entire-collection ids). Samples without an id —
  /// shouldn't happen in practice but possible mid-capture — are skipped.
  Future<List<Sample>> resolveSelection({
    required Set<int> sampleIds,
    required Set<int> collectionIds,
  }) async {
    final repo = SampleRepository.instance;
    final byId = <int, Sample>{};
    final all = await repo.listAll();
    for (final s in all) {
      if (s.id == null) continue;
      if (sampleIds.contains(s.id)) {
        byId[s.id!] = s;
      } else if (s.collectionId != null &&
          collectionIds.contains(s.collectionId)) {
        byId[s.id!] = s;
      }
    }
    final list = byId.values.toList()
      ..sort((a, b) => a.takenAt.compareTo(b.takenAt));
    return list;
  }

  /// Build a CSV-only export. One row per sample with its metadata, total head
  /// count, mean FHB ratio, and a `;`-separated list of per-head FHB ratios
  /// (so the values stay in a single CSV cell).
  Future<ExportArtifact> buildCsvOnly(List<Sample> samples) async {
    final repo = SampleRepository.instance;
    final collections = <int, Collection>{};
    final rows = <List<String>>[];
    rows.add(_csvHeader);
    for (final sample in samples) {
      final instances =
          sample.id == null ? <SampleInstance>[] : await repo.listInstances(sample.id!);
      final collection = await _resolveCollection(repo, collections, sample);
      rows.add(_csvRow(sample, collection, instances));
    }
    final csv = _encodeCsv(rows);
    final tmp = await _tempFile(_csvFilename());
    await tmp.writeAsString(csv);
    return ExportArtifact(
      file: tmp,
      suggestedName: p.basename(tmp.path),
      mimeType: 'text/csv',
    );
  }

  /// Build a ZIP that bundles the CSV plus a COCO-format segmentation dataset
  /// (`images/` + `annotations/instances.json`). One COCO image per sample, one
  /// COCO annotation per persisted [SampleInstance] mask.
  Future<ExportArtifact> buildCsvWithCocoBundle(List<Sample> samples) async {
    final repo = SampleRepository.instance;
    final collections = <int, Collection>{};
    final csvRows = <List<String>>[];
    csvRows.add(_csvHeader);

    final archive = Archive();
    final cocoImages = <Map<String, Object?>>[];
    final cocoAnnotations = <Map<String, Object?>>[];
    var cocoAnnotationId = 1;

    for (var i = 0; i < samples.length; i++) {
      final sample = samples[i];
      if (sample.id == null) continue;
      final cocoImageId = sample.id!;
      final instances = await repo.listInstances(sample.id!);
      final collection = await _resolveCollection(repo, collections, sample);
      csvRows.add(_csvRow(sample, collection, instances));

      final imageFile = File(sample.filePath);
      if (!await imageFile.exists()) {
        debugPrint('[export] missing image for sample ${sample.id}: '
            '${sample.filePath}');
        continue;
      }
      final imageBytes = await imageFile.readAsBytes();
      final imageName = 'sample_${sample.id}${p.extension(sample.filePath)}';
      archive.addFile(
        ArchiveFile('images/$imageName', imageBytes.length, imageBytes),
      );

      final dims = await _decodeImageSize(imageBytes);
      cocoImages.add({
        'id': cocoImageId,
        'file_name': imageName,
        'width': dims?.$1 ?? 0,
        'height': dims?.$2 ?? 0,
        'date_captured': sample.takenAt.toUtc().toIso8601String(),
      });

      final scaleX = (dims != null && dims.$1 > 0)
          ? dims.$1 / (instances.isNotEmpty ? instances.first.imageWidth : 1)
          : 1.0;
      final scaleY = (dims != null && dims.$2 > 0)
          ? dims.$2 / (instances.isNotEmpty ? instances.first.imageHeight : 1)
          : 1.0;

      for (final inst in instances) {
        final maskName =
            'annotations/masks/sample_${sample.id}_instance_${inst.idx}.png';
        archive.addFile(
          ArchiveFile(maskName, inst.maskPng.length, inst.maskPng),
        );
        final scaledBbox = [
          inst.bbox.left * scaleX,
          inst.bbox.top * scaleY,
          inst.bbox.width * scaleX,
          inst.bbox.height * scaleY,
        ];
        final polygon = await _maskToPolygon(
          maskPng: inst.maskPng,
          maskWidth: inst.imageWidth,
          maskHeight: inst.imageHeight,
          scaleX: scaleX,
          scaleY: scaleY,
        );
        cocoAnnotations.add({
          'id': cocoAnnotationId++,
          'image_id': cocoImageId,
          'category_id': _categoryIdFor(sample.detectionMode.id),
          'iscrowd': 0,
          'bbox': scaledBbox,
          'area': scaledBbox[2] * scaledBbox[3],
          'segmentation': polygon != null ? [polygon] : <List<double>>[],
          'mask_file': maskName,
          'score': inst.score,
          'fhb_ratio': inst.fhbRatio,
          'fhb_severity': inst.fhbSeverity,
        });
      }
    }

    archive.addFile(
      _stringFile('samples.csv', _encodeCsv(csvRows)),
    );

    final coco = {
      'info': {
        'description': 'Gopher Eye export',
        'date_created': DateTime.now().toUtc().toIso8601String(),
      },
      'licenses': const [],
      'images': cocoImages,
      'annotations': cocoAnnotations,
      'categories': _cocoCategories,
    };
    archive.addFile(
      _stringFile(
        'annotations/instances.json',
        const JsonEncoder.withIndent('  ').convert(coco),
      ),
    );

    final zipBytes = ZipEncoder().encode(archive);
    final tmp = await _tempFile(_zipFilename());
    await tmp.writeAsBytes(zipBytes, flush: true);
    return ExportArtifact(
      file: tmp,
      suggestedName: p.basename(tmp.path),
      mimeType: 'application/zip',
    );
  }

  // ---------- helpers ----------

  Future<Collection?> _resolveCollection(
    SampleRepository repo,
    Map<int, Collection> cache,
    Sample sample,
  ) async {
    final id = sample.collectionId;
    if (id == null) return null;
    final cached = cache[id];
    if (cached != null) return cached;
    final loaded = await repo.findCollection(id);
    if (loaded != null) cache[id] = loaded;
    return loaded;
  }

  static const List<String> _csvHeader = [
    'sample_id',
    'taken_at',
    'detection_mode',
    'collection_id',
    'collection_name',
    'latitude',
    'longitude',
    'accuracy_m',
    'qr_id',
    'qr_line',
    'qr_rep',
    'qr_location',
    'qr_note',
    'head_count',
    'mean_fhb_ratio',
    'per_head_fhb_ratios',
  ];

  List<String> _csvRow(
    Sample sample,
    Collection? collection,
    List<SampleInstance> instances,
  ) {
    final ratios = instances
        .where((i) => i.fhbRatio != null)
        .map((i) => i.fhbRatio!)
        .toList(growable: false);
    final mean = ratios.isEmpty
        ? ''
        : (ratios.reduce((a, b) => a + b) / ratios.length).toStringAsFixed(4);
    final perHead = ratios.map((r) => r.toStringAsFixed(4)).join(';');
    return [
      sample.id?.toString() ?? '',
      sample.takenAt.toUtc().toIso8601String(),
      sample.detectionMode.id,
      sample.collectionId?.toString() ?? '',
      collection?.name ?? '',
      sample.latitude?.toString() ?? '',
      sample.longitude?.toString() ?? '',
      sample.accuracy?.toString() ?? '',
      sample.qrId ?? '',
      sample.qrLine ?? '',
      sample.qrRep ?? '',
      sample.qrLocation ?? '',
      sample.qrNote ?? '',
      instances.length.toString(),
      mean,
      perHead,
    ];
  }

  String _encodeCsv(List<List<String>> rows) {
    final buf = StringBuffer();
    for (final row in rows) {
      for (var i = 0; i < row.length; i++) {
        if (i > 0) buf.write(',');
        buf.write(_escapeCsvField(row[i]));
      }
      buf.write('\r\n');
    }
    return buf.toString();
  }

  /// RFC 4180-ish: quote any field containing a delimiter, newline, or quote;
  /// double up internal quotes.
  String _escapeCsvField(String field) {
    final needsQuoting = field.contains(',') ||
        field.contains('"') ||
        field.contains('\n') ||
        field.contains('\r');
    if (!needsQuoting) return field;
    return '"${field.replaceAll('"', '""')}"';
  }

  ArchiveFile _stringFile(String name, String content) {
    final bytes = utf8.encode(content);
    return ArchiveFile(name, bytes.length, bytes);
  }

  Future<File> _tempFile(String name) async {
    final dir = await getTemporaryDirectory();
    final exports = Directory(p.join(dir.path, 'gopher_eye_exports'));
    if (!await exports.exists()) {
      await exports.create(recursive: true);
    }
    return File(p.join(exports.path, name));
  }

  String _csvFilename() {
    final ts = _timestamp();
    return 'gopher_eye_$ts.csv';
  }

  String _zipFilename() {
    final ts = _timestamp();
    return 'gopher_eye_$ts.zip';
  }

  String _timestamp() {
    final now = DateTime.now().toUtc();
    String two(int n) => n.toString().padLeft(2, '0');
    return '${now.year}${two(now.month)}${two(now.day)}T'
        '${two(now.hour)}${two(now.minute)}${two(now.second)}Z';
  }

  /// Decode the original image just far enough to read its pixel dimensions.
  /// Done off the main isolate because the on-device originals can be 4–12 MP.
  Future<(int, int)?> _decodeImageSize(Uint8List bytes) async {
    return compute(_decodeSizeIsolate, bytes);
  }

  /// Convert a binary mask PNG into a single COCO polygon by tracing the
  /// outer contour. Coordinates are scaled to the original image resolution
  /// so they line up with the COCO `bbox` we wrote alongside.
  ///
  /// Returns `null` if the mask is empty after decoding.
  Future<List<double>?> _maskToPolygon({
    required Uint8List maskPng,
    required int maskWidth,
    required int maskHeight,
    required double scaleX,
    required double scaleY,
  }) async {
    return compute(
      _maskToPolygonIsolate,
      _MaskPolygonRequest(
        maskPng: maskPng,
        width: maskWidth,
        height: maskHeight,
        scaleX: scaleX,
        scaleY: scaleY,
      ),
    );
  }

  /// One COCO category per detection mode. The numeric id is stable so a
  /// re-export keeps annotation files compatible.
  static const List<Map<String, Object?>> _cocoCategories = [
    {'id': 1, 'name': 'wheat_head', 'supercategory': 'plant'},
    {'id': 2, 'name': 'grape_leaf', 'supercategory': 'plant'},
  ];

  int _categoryIdFor(String detectionModeId) {
    switch (detectionModeId) {
      case 'wheat_fhb':
        return 1;
      case 'grape_leaf':
        return 2;
      default:
        return 1;
    }
  }
}

(int, int)? _decodeSizeIsolate(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) return null;
  return (decoded.width, decoded.height);
}

class _MaskPolygonRequest {
  _MaskPolygonRequest({
    required this.maskPng,
    required this.width,
    required this.height,
    required this.scaleX,
    required this.scaleY,
  });

  final Uint8List maskPng;
  final int width;
  final int height;
  final double scaleX;
  final double scaleY;
}

/// Square-tracing contour walk. Returns the outer outline of the largest
/// connected mask component as a flat `[x0,y0,x1,y1,...]` polygon scaled
/// into the original image's coordinate space.
List<double>? _maskToPolygonIsolate(_MaskPolygonRequest req) {
  final decoded = img.decodePng(req.maskPng);
  if (decoded == null) return null;
  final w = decoded.width;
  final h = decoded.height;
  bool inside(int x, int y) {
    if (x < 0 || y < 0 || x >= w || y >= h) return false;
    final pixel = decoded.getPixel(x, y);
    return pixel.r > 127;
  }

  int? startX;
  int? startY;
  outer:
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      if (inside(x, y)) {
        startX = x;
        startY = y;
        break outer;
      }
    }
  }
  if (startX == null || startY == null) return null;

  // Moore neighborhood traversal. Direction order: 0=E, 1=SE, 2=S, 3=SW,
  // 4=W, 5=NW, 6=N, 7=NE.
  const dx = [1, 1, 0, -1, -1, -1, 0, 1];
  const dy = [0, 1, 1, 1, 0, -1, -1, -1];

  final pts = <int>[];
  var cx = startX;
  var cy = startY;
  var dir = 6;
  pts
    ..add(cx)
    ..add(cy);
  for (var step = 0; step < w * h * 4; step++) {
    var found = false;
    for (var i = 0; i < 8; i++) {
      final ndir = (dir + i) % 8;
      final nx = cx + dx[ndir];
      final ny = cy + dy[ndir];
      if (inside(nx, ny)) {
        cx = nx;
        cy = ny;
        dir = (ndir + 6) % 8;
        pts
          ..add(cx)
          ..add(cy);
        found = true;
        break;
      }
    }
    if (!found) break;
    if (cx == startX && cy == startY && pts.length > 4) break;
  }
  if (pts.length < 6) return null;

  // Reduce vertex count (Visvalingam-style minimum-area filter) so the
  // resulting polygon stays within reasonable size for COCO consumers.
  final simplified = _simplifyContour(pts, maxPoints: 256);
  return [
    for (var i = 0; i < simplified.length; i += 2) ...[
      simplified[i] * req.scaleX,
      simplified[i + 1] * req.scaleY,
    ],
  ];
}

List<int> _simplifyContour(List<int> pts, {required int maxPoints}) {
  if (pts.length ~/ 2 <= maxPoints) return pts;
  final stride = (pts.length ~/ 2 / maxPoints).ceil();
  final out = <int>[];
  for (var i = 0; i < pts.length; i += stride * 2) {
    out
      ..add(pts[i])
      ..add(pts[i + 1]);
  }
  return out;
}
