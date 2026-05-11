import 'dart:io';
import 'dart:ui' show Rect;

import 'package:flutter/foundation.dart';
import 'package:gopher_eye/services/disease_morphology.dart';
import 'package:gopher_eye/services/fhb_pipeline.dart' show FhbReport;
import 'package:gopher_eye/services/wheat_head_pipeline.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

const String _kYoloGrapeAsset = 'assets/models/yolo11_grape_leaf_seg.onnx';
const String _kSwinV2Asset =
    'assets/models/swinv2_grape_leaf_classifier.onnx';

const int _kYoloInputSize = 640;
const int _kSamWorkingEdge = 1024;
const int _kSwinV2InputSize = 224;

/// Number of channels in the grape YOLO output (`[1, 37, 8400]`):
/// `cx, cy, w, h, conf` followed by 32 mask coefficients we discard (SAM
/// produces the masks).
const int _kGrapeYoloChannels = 37;

const double _kScoreThreshold = 0.25;
const double _kNmsIou = 0.5;
const int _kMaxDetections = 100;

/// Grape detections whose mask covers less than this fraction of the working
/// image area are dropped. The user's requirement: "filter out grape leaves
/// that are less than 1/20 of the image size".
const double _kGrapeMinAreaFraction = 1.0 / 20.0;

/// Per-leaf Laplacian-variance focus floor. Crops below this are dropped as
/// out-of-focus before SAM/SwinV2 ever see them. Tuned permissively — phone
/// cameras often hit 200+ on sharp foliage; values < 60 are clearly blurred.
const double _kGrapeFocusVarianceMin = 60.0;

/// Per-leaf disease label, matching the server's grape-leaf SwinV2 labels in
/// `server/app/application.py`:
///   `label2id = {'Healthy-Leaf': 0, 'Downy-Leaf': 1, 'Powdery-Leaf': 2}`.
///
/// The strings flow through `FhbReport.severity` and the existing
/// `fhb_severity` DB column without a schema change.
const String kLabelHealthyLeaf = 'Healthy-Leaf';
const String kLabelDownyLeaf = 'Downy-Leaf';
const String kLabelPowderyLeaf = 'Powdery-Leaf';

/// SwinV2 logit index → label. Matches the server's `id2label` and the
/// classifier ONNX export in `tools/export_swinv2_onnx.py`.
String grapeLeafLabelForIndex(int idx) {
  switch (idx) {
    case 0:
      return kLabelHealthyLeaf;
    case 1:
      return kLabelDownyLeaf;
    case 2:
      return kLabelPowderyLeaf;
    default:
      return kLabelHealthyLeaf;
  }
}

/// User-facing display name for a grape-leaf SwinV2 label. The raw label
/// strings ([kLabelHealthyLeaf] et al.) are pipeline identifiers; this maps
/// them to the names the UI shows in instance captions.
String grapeLeafDisplayLabel(String label) {
  switch (label) {
    case kLabelHealthyLeaf:
      return 'Healthy';
    case kLabelDownyLeaf:
      return 'Downy Mildew';
    case kLabelPowderyLeaf:
      return 'Powdery Mildew';
    default:
      return label;
  }
}

/// Per-leaf grape-leaf disease analyzer.
///
/// Mirrors the segmentation+classifier shape of `server/app/application.py`'s
/// leaf workflow (YOLO leaf segmentation → SwinV2 3-class classifier:
/// Healthy / Downy / Powdery). The label is supplied by the caller after
/// running SwinV2 on the masked crop; this class is purely responsible for
/// turning that label into the per-pixel classification map the disease
/// overlay renderer expects.
///
/// Produces the same [FhbReport] struct as the wheat-FHB analyzer so the rest
/// of the persistence + UI stack (per-instance disease previews, distribution
/// histogram, ratio chip) keeps working unchanged. `green` here means
/// "healthy leaf tissue" and `necrotic` means "diseased tissue" regardless of
/// which disease was the dominant cause; the dominant disease class lives in
/// `severity`.
class GrapeLeafAnalyzer {
  const GrapeLeafAnalyzer();

  static const GrapeLeafAnalyzer instance = GrapeLeafAnalyzer();

  FhbReport analyze({
    required img.Image workingImage,
    required Uint8List leafMask,
    required int maskWidth,
    required int maskHeight,
    required Rect bbox,
    required String label,
  }) {
    if (workingImage.width != maskWidth || workingImage.height != maskHeight) {
      throw ArgumentError(
        'workingImage size (${workingImage.width}×${workingImage.height}) '
        'must equal mask size ($maskWidth×$maskHeight)',
      );
    }

    final classification = Uint8List(maskWidth * maskHeight);
    final fillClass = _classForLabel(label);

    final x0 = bbox.left.floor().clamp(0, maskWidth - 1);
    final y0 = bbox.top.floor().clamp(0, maskHeight - 1);
    final x1 = bbox.right.ceil().clamp(x0 + 1, maskWidth);
    final y1 = bbox.bottom.ceil().clamp(y0 + 1, maskHeight);

    int totalPixels = 0;
    for (int y = y0; y < y1; y++) {
      final row = y * maskWidth;
      for (int x = x0; x < x1; x++) {
        final i = row + x;
        if (leafMask[i] == 0) continue;
        classification[i] = fillClass;
        totalPixels++;
      }
    }

    final isHealthy = label == kLabelHealthyLeaf;
    final greenCount = isHealthy ? totalPixels : 0;
    final necroticCount = isHealthy ? 0 : totalPixels;
    final ratio = isHealthy ? 0.0 : (totalPixels == 0 ? 0.0 : 1.0);

    return FhbReport(
      greenCount: greenCount,
      necroticCount: necroticCount,
      otherCount: 0,
      totalPixels: totalPixels,
      fhbRatio: ratio,
      severity: label,
      classification: classification,
      maskWidth: maskWidth,
      maskHeight: maskHeight,
    );
  }

  /// Map a SwinV2 label to the per-pixel class id that drives the overlay
  /// colours: healthy → green tint, downy → red, powdery → yellow.
  int _classForLabel(String label) {
    switch (label) {
      case kLabelHealthyLeaf:
        return kClassGreen;
      case kLabelDownyLeaf:
        return kClassNecrotic;
      case kLabelPowderyLeaf:
        return kClassOther;
      default:
        return kClassGreen;
    }
  }
}

/// On-device grape-leaf detection + classification pipeline.
///
///   YOLO11-seg locates leaf bboxes (its mask coefficients are discarded —
///   SAM produces tighter masks) → SAM refines each bbox into a per-leaf
///   mask using a single global encoder pass over the working image plus
///   per-detection point+box prompts (same path the instance editor uses) →
///   leaves smaller than [_kGrapeMinAreaFraction] of the working image, or
///   below the Laplacian-variance focus floor [_kGrapeFocusVarianceMin], are
///   dropped → SwinV2 classifies the surviving masked crops as Healthy /
///   Downy / Powdery.
///
/// SAM sessions live on [WheatHeadPipeline] so the editor and other
/// SAM-using callers don't double-load the encoder. Only the grape YOLO
/// and the SwinV2 classifier are owned here.
class GrapeLeafPipeline {
  GrapeLeafPipeline._();
  static final GrapeLeafPipeline instance = GrapeLeafPipeline._();

  OrtSession? _yolo;
  OrtSession? _swinV2;
  Future<void>? _initFuture;

  /// Lazy-load grape YOLO + SwinV2 (107 MB combined). Memoised after the
  /// first call so repeated detections only pay the cost once.
  Future<void> _ensureLoaded() => _initFuture ??= _load();

  Future<void> _load() async {
    // SAM sessions belong to the wheat pipeline; ensure they're up before we
    // try to refine any boxes.
    await WheatHeadPipeline.instance.ensureSam();
    final opts = OrtSessionOptions();
    try {
      final yoloBytes = await WheatHeadPipeline.loadAsset(_kYoloGrapeAsset);
      final swinBytes = await WheatHeadPipeline.loadAsset(_kSwinV2Asset);
      _yolo = OrtSession.fromBuffer(yoloBytes, opts);
      _swinV2 = OrtSession.fromBuffer(swinBytes, opts);
    } finally {
      opts.release();
    }
  }

  /// Runs the grape-leaf detection pipeline on [imageFile]. Returns a
  /// [WheatHeadResult] (the type is a misnomer at this point — its shape
  /// generalises to any per-instance detector and the rest of the
  /// persistence/UI stack consumes it directly). SwinV2 classification is
  /// deferred to [analyzeDisease] so that re-running disease analysis after
  /// a mask edit re-classifies through the same code path.
  Future<WheatHeadResult> run(
    File imageFile, {
    void Function(int done, int total)? onProgress,
  }) async {
    final stopwatch = Stopwatch()..start();
    await _ensureLoaded();

    final raw = await imageFile.readAsBytes();
    final fullRes = img.decodeImage(raw);
    if (fullRes == null) {
      throw StateError('Could not decode image: ${imageFile.path}');
    }

    final wheat = WheatHeadPipeline.instance;
    final working = wheat.resizeLongestEdge(fullRes, _kSamWorkingEdge);
    final w = working.width;
    final h = working.height;
    final imageArea = w * h;
    final minMaskArea = (imageArea * _kGrapeMinAreaFraction).round();

    final candidates = await _detectLeaves(working);
    debugPrint('[grape] yolo found ${candidates.length} candidates');
    onProgress?.call(0, candidates.length);

    final embedding = await wheat.prepareEditFromImage(working);
    final detections = <WheatHeadDetection>[];
    int dropSize = 0;
    int dropFocus = 0;
    try {
      for (int i = 0; i < candidates.length; i++) {
        final cand = candidates[i];
        final mask = await wheat.predict(
          embedding: embedding,
          origW: w,
          origH: h,
          points: [cand.centroid],
          pointLabels: [1],
          bbox: cand.bbox,
        );

        // (a) Size filter: drop leaves whose mask area is < image_area / 20.
        int maskArea = 0;
        for (int p = 0; p < mask.length; p++) {
          if (mask[p] != 0) maskArea++;
        }
        if (maskArea < minMaskArea) {
          dropSize++;
          onProgress?.call(i + 1, candidates.length);
          continue;
        }

        // (b) Focus filter: Laplacian variance over the masked leaf region of
        //     the working image. Out-of-focus crops produce flat Laplacian
        //     responses (low variance).
        final variance = _laplacianVarianceMasked(
          image: working,
          mask: mask,
          bbox: cand.bbox,
          maskWidth: w,
          maskHeight: h,
        );
        if (variance < _kGrapeFocusVarianceMin) {
          dropFocus++;
          onProgress?.call(i + 1, candidates.length);
          continue;
        }

        detections.add(WheatHeadDetection(
          bbox: cand.bbox,
          centroid: cand.centroid,
          score: cand.score,
          mask: mask,
        ));
        onProgress?.call(i + 1, candidates.length);
      }
    } finally {
      embedding.release();
    }
    debugPrint(
      '[grape] kept ${detections.length} '
      '(dropped $dropSize too-small, $dropFocus blurry)',
    );

    final imagePng = Uint8List.fromList(img.encodePng(working));
    final overlayPng = wheat.renderInstanceOverlay(detections, w, h);
    stopwatch.stop();
    return WheatHeadResult(
      imagePng: imagePng,
      overlayPng: overlayPng,
      width: w,
      height: h,
      detections: detections,
      elapsed: stopwatch.elapsed,
    );
  }

  /// SwinV2-driven disease analysis for a single leaf instance. Mirrors
  /// [WheatHeadPipeline.analyzeDisease]'s shape so the detection service
  /// can dispatch by mode.
  Future<FhbReport> analyzeDisease({
    required img.Image workingImage,
    required Uint8List mask,
    required int maskWidth,
    required int maskHeight,
    required Rect bbox,
  }) async {
    await _ensureLoaded();
    final label = await _classifyLeaf(
      workingImage: workingImage,
      mask: mask,
      maskWidth: maskWidth,
      maskHeight: maskHeight,
      bbox: bbox,
    );
    return GrapeLeafAnalyzer.instance.analyze(
      workingImage: workingImage,
      leafMask: mask,
      maskWidth: maskWidth,
      maskHeight: maskHeight,
      bbox: bbox,
      label: label,
    );
  }

  // ---------- YOLO11-seg (pre-NMS, channel-major) ----------

  Future<List<Candidate>> _detectLeaves(img.Image working) async {
    final session = _yolo!;
    final wheat = WheatHeadPipeline.instance;
    final letterboxed = wheat.letterbox(working, _kYoloInputSize);
    final input = wheat.imageToYoloTensor(letterboxed.image);
    final tensor = OrtValueTensor.createTensorWithDataList(
      input,
      [1, 3, _kYoloInputSize, _kYoloInputSize],
    );
    final yoloInputs = session.inputNames;
    if (yoloInputs.isEmpty) {
      tensor.release();
      throw StateError('Grape YOLO ONNX session has no input names');
    }
    final inputName = yoloInputs.first;
    final runOpts = OrtRunOptions();
    final outputs = await session.runAsync(runOpts, {inputName: tensor});
    tensor.release();
    runOpts.release();
    if (outputs == null || outputs.isEmpty || outputs.first == null) {
      return const [];
    }
    final raw = outputs.first!.value;
    for (final o in outputs) {
      o?.release();
    }
    return _parseYoloOutput(
      raw,
      letterbox: letterboxed,
      origW: working.width,
      origH: working.height,
    );
  }

  /// Parse the grape YOLO11-seg pre-NMS output (channel-major
  /// `[1, 37, 8400]`):
  ///   ch 0..3 → cx, cy, w, h (640-letterbox px)
  ///   ch 4    → class-0 confidence
  ///   ch 5..36 → 32 mask coeffs (discarded; SAM produces the masks)
  List<Candidate> _parseYoloOutput(
    Object? raw, {
    required LetterboxResult letterbox,
    required int origW,
    required int origH,
  }) {
    final wheat = WheatHeadPipeline.instance;
    final flat = wheat.flattenToDoubles(raw);
    if (flat.isEmpty) return const [];
    if (flat.length % _kGrapeYoloChannels != 0) {
      debugPrint('[grape] unexpected yolo output length ${flat.length}');
      return const [];
    }
    final n = flat.length ~/ _kGrapeYoloChannels;

    final candidates = <Candidate>[];
    for (int a = 0; a < n; a++) {
      final score = flat[4 * n + a];
      if (score < _kScoreThreshold) continue;
      candidates.add(wheat.remap(
        cx: flat[0 * n + a],
        cy: flat[1 * n + a],
        w: flat[2 * n + a],
        h: flat[3 * n + a],
        score: score,
        letterbox: letterbox,
        origW: origW,
        origH: origH,
        xyxy: false,
      ));
    }

    candidates.sort((a, b) => b.score.compareTo(a.score));
    final kept = wheat.nms(candidates, _kNmsIou);
    if (kept.length > _kMaxDetections) {
      return kept.sublist(0, _kMaxDetections);
    }
    return kept;
  }

  // ---------- SwinV2 grape-leaf classifier ----------

  /// Classify a single leaf with the SwinV2 3-class head. Mirrors the
  /// server's `Classification.classify`: zero out background pixels, resize
  /// the masked crop to 224×224, normalise to `[0, 1]` (no ImageNet mean/std
  /// — `transforms.functional.to_tensor` only divides by 255), argmax the
  /// returned logits.
  Future<String> _classifyLeaf({
    required img.Image workingImage,
    required Uint8List mask,
    required int maskWidth,
    required int maskHeight,
    required Rect bbox,
  }) async {
    final session = _swinV2!;

    final x0 = bbox.left.floor().clamp(0, maskWidth - 1);
    final y0 = bbox.top.floor().clamp(0, maskHeight - 1);
    final x1 = bbox.right.ceil().clamp(x0 + 1, maskWidth);
    final y1 = bbox.bottom.ceil().clamp(y0 + 1, maskHeight);
    final cropW = x1 - x0;
    final cropH = y1 - y0;

    // Build the masked crop at native resolution. Pixels outside the mask
    // are zeroed so the classifier sees the same input as the server.
    final cropped =
        img.Image(width: cropW, height: cropH, numChannels: 3);
    final src = workingImage.getBytes(order: img.ChannelOrder.rgb);
    for (int y = 0; y < cropH; y++) {
      for (int x = 0; x < cropW; x++) {
        final gx = x + x0;
        final gy = y + y0;
        if (mask[gy * maskWidth + gx] == 0) {
          cropped.setPixelRgb(x, y, 0, 0, 0);
          continue;
        }
        final p = (gy * maskWidth + gx) * 3;
        cropped.setPixelRgb(x, y, src[p], src[p + 1], src[p + 2]);
      }
    }

    final resized = img.copyResize(
      cropped,
      width: _kSwinV2InputSize,
      height: _kSwinV2InputSize,
      interpolation: img.Interpolation.linear,
    );

    final tensorData = Float32List(3 * _kSwinV2InputSize * _kSwinV2InputSize);
    const plane = _kSwinV2InputSize * _kSwinV2InputSize;
    for (int y = 0; y < _kSwinV2InputSize; y++) {
      for (int x = 0; x < _kSwinV2InputSize; x++) {
        final px = resized.getPixel(x, y);
        final i = y * _kSwinV2InputSize + x;
        tensorData[i] = px.r / 255.0;
        tensorData[plane + i] = px.g / 255.0;
        tensorData[2 * plane + i] = px.b / 255.0;
      }
    }

    final tensor = OrtValueTensor.createTensorWithDataList(
      tensorData,
      [1, 3, _kSwinV2InputSize, _kSwinV2InputSize],
    );
    final swinInputs = session.inputNames;
    if (swinInputs.isEmpty) {
      tensor.release();
      throw StateError('SwinV2 ONNX session has no input names');
    }
    final runOpts = OrtRunOptions();
    final outputs = await session
        .runAsync(runOpts, {swinInputs.first: tensor});
    tensor.release();
    runOpts.release();
    if (outputs == null || outputs.isEmpty || outputs.first == null) {
      throw StateError('SwinV2 returned no output');
    }
    final logits =
        WheatHeadPipeline.instance.flattenToDoubles(outputs.first!.value);
    for (final o in outputs) {
      o?.release();
    }

    int bestIdx = 0;
    double bestVal = logits.isEmpty ? 0 : logits[0];
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        bestIdx = i;
      }
    }
    return grapeLeafLabelForIndex(bestIdx);
  }

  // ---------- Focus check ----------

  /// Variance of the Laplacian over the masked region of [image]. Standard
  /// CV blur metric: a sharp leaf has high-frequency edges that produce a
  /// Laplacian response with high variance; a defocused leaf is locally
  /// flat and the variance collapses. We restrict the computation to mask
  /// interior pixels (4-neighbours all inside the mask) so the kernel never
  /// straddles the leaf boundary or the zero background.
  double _laplacianVarianceMasked({
    required img.Image image,
    required Uint8List mask,
    required Rect bbox,
    required int maskWidth,
    required int maskHeight,
  }) {
    final x0 = bbox.left.floor().clamp(1, maskWidth - 2);
    final y0 = bbox.top.floor().clamp(1, maskHeight - 2);
    final x1 = bbox.right.ceil().clamp(x0 + 1, maskWidth - 1);
    final y1 = bbox.bottom.ceil().clamp(y0 + 1, maskHeight - 1);
    final src = image.getBytes(order: img.ChannelOrder.rgb);

    int gray(int x, int y) {
      final p = (y * maskWidth + x) * 3;
      // Rec. 601 luma, integer-only — close enough for a relative metric.
      return (src[p] * 299 + src[p + 1] * 587 + src[p + 2] * 114) ~/ 1000;
    }

    double sum = 0;
    double sumSq = 0;
    int count = 0;
    for (int y = y0; y < y1; y++) {
      final row = y * maskWidth;
      for (int x = x0; x < x1; x++) {
        if (mask[row + x] == 0) continue;
        if (mask[row + x - 1] == 0 ||
            mask[row + x + 1] == 0 ||
            mask[row + x - maskWidth] == 0 ||
            mask[row + x + maskWidth] == 0) {
          continue;
        }
        final lap = (gray(x - 1, y) +
                gray(x + 1, y) +
                gray(x, y - 1) +
                gray(x, y + 1) -
                4 * gray(x, y))
            .toDouble();
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }
    if (count < 32) return 0; // too few interior samples to trust the metric
    final mean = sum / count;
    final variance = (sumSq / count) - (mean * mean);
    return variance < 0 ? 0 : variance;
  }
}
