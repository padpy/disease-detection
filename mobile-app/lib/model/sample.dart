import 'package:gopher_eye/model/detection_mode.dart';

class Sample {
  const Sample({
    this.id,
    required this.filePath,
    required this.takenAt,
    this.latitude,
    this.longitude,
    this.accuracy,
    this.detectionMode = DetectionMode.wheatFhb,
    this.collectionId,
    this.qrId,
    this.qrLine,
    this.qrRep,
    this.qrLocation,
    this.qrNote,
  });

  final int? id;
  final String filePath;
  final DateTime takenAt;
  final double? latitude;
  final double? longitude;
  final double? accuracy;

  /// Optional grouping. Set on capture when a collection is active in the
  /// camera screen; nullable so legacy samples (and "None" sessions) don't
  /// belong to a collection. Can be reassigned later from the inspector.
  final int? collectionId;

  /// Which on-device pipeline this sample is processed with. Set on capture
  /// (from the camera screen's mode picker) and may be changed later from the
  /// sample viewer. Determines which classifier the disease analyzer runs.
  final DetectionMode detectionMode;

  /// Optional metadata sourced from a Gopher Eye QR code scanned just before
  /// capture. All fields are nullable — a sample may have any combination
  /// (e.g. a plot id but no rep number).
  final String? qrId;
  final String? qrLine;
  final String? qrRep;
  final String? qrLocation;
  final String? qrNote;

  bool get hasLocation => latitude != null && longitude != null;

  bool get hasQrMetadata =>
      _notBlank(qrId) ||
      _notBlank(qrLine) ||
      _notBlank(qrRep) ||
      _notBlank(qrLocation) ||
      _notBlank(qrNote);

  Sample copyWith({
    int? id,
    DetectionMode? detectionMode,
    Object? collectionId = _unset,
    String? qrId,
    String? qrLine,
    String? qrRep,
    String? qrLocation,
    String? qrNote,
  }) =>
      Sample(
        id: id ?? this.id,
        filePath: filePath,
        takenAt: takenAt,
        latitude: latitude,
        longitude: longitude,
        accuracy: accuracy,
        detectionMode: detectionMode ?? this.detectionMode,
        collectionId: identical(collectionId, _unset)
            ? this.collectionId
            : collectionId as int?,
        qrId: qrId ?? this.qrId,
        qrLine: qrLine ?? this.qrLine,
        qrRep: qrRep ?? this.qrRep,
        qrLocation: qrLocation ?? this.qrLocation,
        qrNote: qrNote ?? this.qrNote,
      );

  Map<String, Object?> toMap() => {
        if (id != null) 'id': id,
        'file_path': filePath,
        'taken_at': takenAt.millisecondsSinceEpoch,
        'latitude': latitude,
        'longitude': longitude,
        'accuracy': accuracy,
        'detection_mode': detectionMode.id,
        'collection_id': collectionId,
        'qr_id': qrId,
        'qr_line': qrLine,
        'qr_rep': qrRep,
        'qr_location': qrLocation,
        'qr_note': qrNote,
      };

  factory Sample.fromMap(Map<String, Object?> row) => Sample(
        id: row['id'] as int?,
        filePath: row['file_path'] as String,
        takenAt: DateTime.fromMillisecondsSinceEpoch(row['taken_at'] as int),
        latitude: row['latitude'] as double?,
        longitude: row['longitude'] as double?,
        accuracy: row['accuracy'] as double?,
        detectionMode: DetectionMode.fromId(row['detection_mode'] as String?),
        collectionId: row['collection_id'] as int?,
        qrId: row['qr_id'] as String?,
        qrLine: row['qr_line'] as String?,
        qrRep: row['qr_rep'] as String?,
        qrLocation: row['qr_location'] as String?,
        qrNote: row['qr_note'] as String?,
      );

  static bool _notBlank(String? v) => v != null && v.isNotEmpty;

  /// Sentinel for `copyWith` so callers can pass `null` to clear a field
  /// (e.g. removing a sample from its collection) without colliding with
  /// "no change" semantics.
  static const Object _unset = Object();
}
