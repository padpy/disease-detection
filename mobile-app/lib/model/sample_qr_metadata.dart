import 'dart:convert';

/// Metadata encoded into / decoded from a Gopher Eye QR code. Used to attach
/// human-readable identifiers and notes to a capture so field samples can be
/// correlated with plot/site records without manual entry on the device.
///
/// The on-the-wire format is JSON tagged with [_payloadTag] so we don't try to
/// interpret arbitrary QR codes as our own. Plain JSON without the tag is
/// still accepted (lenient parsing) so users can hand-craft codes.
class SampleQrMetadata {
  const SampleQrMetadata({
    this.qrId,
    this.line,
    this.rep,
    this.location,
    this.note,
  });

  final String? qrId;
  final String? line;
  final String? rep;
  final String? location;
  final String? note;

  static const _payloadTag = 'gopher_eye_sample';

  bool get isEmpty =>
      _isBlank(qrId) &&
      _isBlank(line) &&
      _isBlank(rep) &&
      _isBlank(location) &&
      _isBlank(note);

  /// Short label suitable for chips ("id · line · rep"). Falls back to
  /// whichever fields are set; returns null when nothing is populated.
  String? get displayLabel {
    final parts = <String>[];
    if (!_isBlank(qrId)) parts.add(qrId!);
    if (!_isBlank(line)) parts.add(line!);
    if (!_isBlank(rep)) parts.add('rep ${rep!}');
    if (parts.isEmpty && !_isBlank(location)) parts.add(location!);
    if (parts.isEmpty) return null;
    return parts.join(' · ');
  }

  String encode() {
    final payload = <String, Object?>{
      'type': _payloadTag,
      if (!_isBlank(qrId)) 'id': qrId,
      if (!_isBlank(line)) 'line': line,
      if (!_isBlank(rep)) 'rep': rep,
      if (!_isBlank(location)) 'location': location,
      if (!_isBlank(note)) 'note': note,
    };
    return jsonEncode(payload);
  }

  /// Best-effort parse of [raw]. Returns null on any decoding failure or when
  /// the result would be empty. Accepts both tagged Gopher Eye payloads and
  /// plain JSON objects with the same field names.
  static SampleQrMetadata? tryDecode(String raw) {
    final trimmed = raw.trim();
    if (trimmed.isEmpty) return null;
    try {
      final decoded = jsonDecode(trimmed);
      if (decoded is! Map) return null;
      final map = decoded.cast<String, Object?>();
      final meta = SampleQrMetadata(
        qrId: _str(map['id']) ?? _str(map['i']),
        line: _str(map['line']),
        rep: _str(map['rep']),
        location: _str(map['location']) ?? _str(map['loc']),
        note: _str(map['note']),
      );
      return meta.isEmpty ? null : meta;
    } catch (_) {
      return null;
    }
  }

  SampleQrMetadata copyWith({
    String? qrId,
    String? line,
    String? rep,
    String? location,
    String? note,
  }) =>
      SampleQrMetadata(
        qrId: qrId ?? this.qrId,
        line: line ?? this.line,
        rep: rep ?? this.rep,
        location: location ?? this.location,
        note: note ?? this.note,
      );

  static bool _isBlank(String? v) => v == null || v.isEmpty;

  static String? _str(Object? v) {
    if (v == null) return null;
    final s = v.toString().trim();
    return s.isEmpty ? null : s;
  }
}
