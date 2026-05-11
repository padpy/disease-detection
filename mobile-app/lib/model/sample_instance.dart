import 'dart:typed_data';
import 'dart:ui' show Offset, Rect;

class SampleInstance {
  const SampleInstance({
    this.id,
    required this.sampleId,
    required this.idx,
    required this.bbox,
    required this.centroid,
    required this.score,
    required this.imageWidth,
    required this.imageHeight,
    required this.maskPng,
    required this.previewPng,
    required this.createdAt,
    required this.updatedAt,
    this.fhbGreenCount,
    this.fhbNecroticCount,
    this.fhbOtherCount,
    this.fhbTotalPixels,
    this.fhbRatio,
    this.fhbSeverity,
    this.diseasePreviewPng,
  });

  final int? id;
  final int sampleId;
  final int idx;
  final Rect bbox;
  final Offset centroid;
  final double score;

  /// Working-image dimensions that [bbox], [centroid] and the decoded [maskPng]
  /// reference. Stored alongside the mask so future edits can map correctly
  /// even if the source file is re-decoded at a different resolution.
  final int imageWidth;
  final int imageHeight;

  /// Single-channel PNG of the binary mask (255 = inside, 0 = outside),
  /// sized [imageWidth] × [imageHeight].
  final Uint8List maskPng;

  /// RGBA PNG of the instance preview: cropped working image around [bbox]
  /// with padding plus the segmentation outline drawn on top.
  final Uint8List previewPng;

  final DateTime createdAt;
  final DateTime updatedAt;

  // ---------- Disease detection (FHB) ----------

  /// Healthy-pixel count after HSV classification + morphological closure +
  /// small-contour filter. `null` when disease analysis hasn't been run for
  /// this instance yet.
  final int? fhbGreenCount;

  /// Necrotic (FHB) pixel count after the same cleanup steps.
  final int? fhbNecroticCount;

  /// Pixels inside the mask that didn't pass either HSV gate.
  final int? fhbOtherCount;

  /// Total pixels inside the spike mask.
  final int? fhbTotalPixels;

  /// FHB% expressed as `necrotic / (necrotic + green)`. The chart uses this
  /// value directly.
  final double? fhbRatio;

  /// Severity bucket label for quick display (Healthy / Mild FHB / …).
  final String? fhbSeverity;

  /// Cropped RGBA preview tile showing the per-pixel disease classification
  /// (green / red / yellow). Same dimensions as [previewPng].
  final Uint8List? diseasePreviewPng;

  bool get hasDiseaseAnalysis => fhbRatio != null;

  SampleInstance copyWith({
    int? id,
    Rect? bbox,
    Offset? centroid,
    Uint8List? maskPng,
    Uint8List? previewPng,
    DateTime? updatedAt,
    int? fhbGreenCount,
    int? fhbNecroticCount,
    int? fhbOtherCount,
    int? fhbTotalPixels,
    double? fhbRatio,
    String? fhbSeverity,
    Uint8List? diseasePreviewPng,
    bool clearDiseaseAnalysis = false,
  }) {
    return SampleInstance(
      id: id ?? this.id,
      sampleId: sampleId,
      idx: idx,
      bbox: bbox ?? this.bbox,
      centroid: centroid ?? this.centroid,
      score: score,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      maskPng: maskPng ?? this.maskPng,
      previewPng: previewPng ?? this.previewPng,
      createdAt: createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
      fhbGreenCount:
          clearDiseaseAnalysis ? null : (fhbGreenCount ?? this.fhbGreenCount),
      fhbNecroticCount: clearDiseaseAnalysis
          ? null
          : (fhbNecroticCount ?? this.fhbNecroticCount),
      fhbOtherCount:
          clearDiseaseAnalysis ? null : (fhbOtherCount ?? this.fhbOtherCount),
      fhbTotalPixels: clearDiseaseAnalysis
          ? null
          : (fhbTotalPixels ?? this.fhbTotalPixels),
      fhbRatio: clearDiseaseAnalysis ? null : (fhbRatio ?? this.fhbRatio),
      fhbSeverity:
          clearDiseaseAnalysis ? null : (fhbSeverity ?? this.fhbSeverity),
      diseasePreviewPng: clearDiseaseAnalysis
          ? null
          : (diseasePreviewPng ?? this.diseasePreviewPng),
    );
  }

  Map<String, Object?> toMap() => {
        if (id != null) 'id': id,
        'sample_id': sampleId,
        'idx': idx,
        'bbox_left': bbox.left,
        'bbox_top': bbox.top,
        'bbox_right': bbox.right,
        'bbox_bottom': bbox.bottom,
        'centroid_x': centroid.dx,
        'centroid_y': centroid.dy,
        'score': score,
        'image_w': imageWidth,
        'image_h': imageHeight,
        'mask_png': maskPng,
        'preview_png': previewPng,
        'created_at': createdAt.millisecondsSinceEpoch,
        'updated_at': updatedAt.millisecondsSinceEpoch,
        'fhb_green': fhbGreenCount,
        'fhb_necrotic': fhbNecroticCount,
        'fhb_other': fhbOtherCount,
        'fhb_total': fhbTotalPixels,
        'fhb_ratio': fhbRatio,
        'fhb_severity': fhbSeverity,
        'disease_preview_png': diseasePreviewPng,
      };

  factory SampleInstance.fromMap(Map<String, Object?> row) => SampleInstance(
        id: row['id'] as int?,
        sampleId: row['sample_id'] as int,
        idx: row['idx'] as int,
        bbox: Rect.fromLTRB(
          (row['bbox_left'] as num).toDouble(),
          (row['bbox_top'] as num).toDouble(),
          (row['bbox_right'] as num).toDouble(),
          (row['bbox_bottom'] as num).toDouble(),
        ),
        centroid: Offset(
          (row['centroid_x'] as num).toDouble(),
          (row['centroid_y'] as num).toDouble(),
        ),
        score: (row['score'] as num).toDouble(),
        imageWidth: row['image_w'] as int,
        imageHeight: row['image_h'] as int,
        maskPng: row['mask_png'] as Uint8List,
        previewPng: row['preview_png'] as Uint8List,
        createdAt:
            DateTime.fromMillisecondsSinceEpoch(row['created_at'] as int),
        updatedAt:
            DateTime.fromMillisecondsSinceEpoch(row['updated_at'] as int),
        fhbGreenCount: row['fhb_green'] as int?,
        fhbNecroticCount: row['fhb_necrotic'] as int?,
        fhbOtherCount: row['fhb_other'] as int?,
        fhbTotalPixels: row['fhb_total'] as int?,
        fhbRatio: (row['fhb_ratio'] as num?)?.toDouble(),
        fhbSeverity: row['fhb_severity'] as String?,
        diseasePreviewPng: row['disease_preview_png'] as Uint8List?,
      );
}
