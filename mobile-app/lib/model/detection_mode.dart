/// What kind of plant the sample shows, and therefore which on-device pipeline
/// runs against it.
///
/// Stored per-sample so the same app session can capture both wheat-head
/// (FHB) imagery and grape-leaf imagery without losing track. The default for
/// legacy samples (captured before the mode column existed) is [wheatFhb].
enum DetectionMode {
  /// Wheat-head FHB pipeline: YOLO detects spikes, SAM segments each, the FHB
  /// HSV classifier produces necrotic / healthy ratios.
  wheatFhb,

  /// Grape-leaf disease pipeline: YOLO11-seg locates leaves, SAM refines each
  /// bbox into a tight mask, then a SwinV2 classifier (Healthy / Downy /
  /// Powdery) labels the masked crop. Leaves smaller than 1/20 of the working
  /// image, or below the focus threshold, are dropped.
  grapeLeaf;

  /// Stable identifier persisted in sqlite. Don't rename without a migration.
  String get id => switch (this) {
        DetectionMode.wheatFhb => 'wheat_fhb',
        DetectionMode.grapeLeaf => 'grape_leaf',
      };

  /// Short label for chips and badges.
  String get label => switch (this) {
        DetectionMode.wheatFhb => 'FHB',
        DetectionMode.grapeLeaf => 'Grape Leaf',
      };

  /// Long label for menus and dialogs.
  String get longLabel => switch (this) {
        DetectionMode.wheatFhb => 'Wheat — Fusarium head blight',
        DetectionMode.grapeLeaf => 'Grape leaf disease',
      };

  /// Singular noun for an instance under this mode.
  String get instanceNounSingular => switch (this) {
        DetectionMode.wheatFhb => 'wheat head',
        DetectionMode.grapeLeaf => 'leaf',
      };

  String get instanceNounPlural => switch (this) {
        DetectionMode.wheatFhb => 'wheat heads',
        DetectionMode.grapeLeaf => 'leaves',
      };

  /// Whether the on-device pipeline includes an automatic instance detector
  /// (YOLO). Modes without one rely on the manual instance editor.
  bool get hasAutoDetection => switch (this) {
        DetectionMode.wheatFhb => true,
        DetectionMode.grapeLeaf => true,
      };

  /// Title shown above the per-sample distribution histogram.
  String get distributionChartTitle => switch (this) {
        DetectionMode.wheatFhb => 'FHB% distribution',
        DetectionMode.grapeLeaf => 'Disease% distribution',
      };

  static DetectionMode fromId(String? id) {
    for (final m in DetectionMode.values) {
      if (m.id == id) return m;
    }
    return DetectionMode.wheatFhb;
  }
}
