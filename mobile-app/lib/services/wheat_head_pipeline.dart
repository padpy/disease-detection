import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' show Rect, Offset;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:gopher_eye/services/fhb_pipeline.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

const String _kYoloAsset = 'assets/models/yolo26_wheat_head.onnx';
const String _kSamEncoderAsset = 'assets/models/sam3_efficient_encoder.onnx';
const String _kSamDecoderAsset = 'assets/models/sam3_efficient_decoder.onnx';

const int _kYoloInputSize = 640;
const int _kSamInputSize = 1024;

const double _kScoreThreshold = 0.25;
const double _kNmsIou = 0.5;
const int _kMaxDetections = 100;

const List<double> _kImagenetMean = [0.485, 0.456, 0.406];
const List<double> _kImagenetStd = [0.229, 0.224, 0.225];

class WheatHeadDetection {
  const WheatHeadDetection({
    required this.bbox,
    required this.centroid,
    required this.score,
    required this.mask,
  });

  /// In pixels of the working (downscaled) image.
  final Rect bbox;
  final Offset centroid;
  final double score;

  /// Binary mask sized [WheatHeadResult.height] x [WheatHeadResult.width],
  /// row-major; 1 = wheat head, 0 = background.
  final Uint8List mask;
}

class WheatHeadResult {
  const WheatHeadResult({
    required this.imagePng,
    required this.overlayPng,
    required this.width,
    required this.height,
    required this.detections,
    required this.elapsed,
  });

  /// PNG of the working (downscaled) source image.
  final Uint8List imagePng;

  /// PNG of an RGBA overlay (transparent background, coloured masks +
  /// centroids). Same size as [imagePng].
  final Uint8List overlayPng;
  final int width;
  final int height;
  final List<WheatHeadDetection> detections;
  final Duration elapsed;
}

/// Wrapper around a SAM image embedding plus the letterbox state used to
/// produce it. Caller owns the lifetime — call [release] to free the
/// underlying ORT tensor.
class EditableEmbedding {
  EditableEmbedding._(this._embedding);
  final _Embedding _embedding;

  void release() => _embedding.release();
}

/// On-device pipeline: YOLO26 for wheat-head centroids → SAM-efficient for
/// per-head masks. Sessions are created lazily on first use and kept alive
/// for the rest of the process — the encoder asset is ~27 MB and the YOLO
/// model is similar, so reloading per frame would be wasteful.
class WheatHeadPipeline {
  WheatHeadPipeline._();
  static final WheatHeadPipeline instance = WheatHeadPipeline._();

  bool _envInitialised = false;
  OrtSession? _yolo;
  OrtSession? _samEncoder;
  OrtSession? _samDecoder;

  /// Each future is created once on first request, then memoised. The SAM
  /// stack is split out so the instance editor and other on-device pipelines
  /// (e.g. [GrapeLeafPipeline]) can re-use the encoder/decoder without
  /// paying for a wheat YOLO load.
  Future<void>? _samFuture;
  Future<void>? _wheatFuture;

  Future<void> ensureSam() => _samFuture ??= _loadSam();
  Future<void> _ensureWheatLoaded() =>
      _wheatFuture ??= _loadWheatStack();

  Future<void> _loadSam() async {
    if (!_envInitialised) {
      OrtEnv.instance.init();
      _envInitialised = true;
    }
    final opts = OrtSessionOptions();
    try {
      final encBytes = await loadAsset(_kSamEncoderAsset);
      final decBytes = await loadAsset(_kSamDecoderAsset);
      _samEncoder = OrtSession.fromBuffer(encBytes, opts);
      _samDecoder = OrtSession.fromBuffer(decBytes, opts);
    } finally {
      opts.release();
    }
  }

  Future<void> _loadWheatStack() async {
    await ensureSam();
    final opts = OrtSessionOptions();
    try {
      final yoloBytes = await loadAsset(_kYoloAsset);
      _yolo = OrtSession.fromBuffer(yoloBytes, opts);
    } finally {
      opts.release();
    }
  }

  /// Public asset loader shared with [GrapeLeafPipeline] so it doesn't have
  /// to duplicate the trivial `rootBundle.load` boilerplate.
  static Future<Uint8List> loadAsset(String path) async {
    final data = await rootBundle.load(path);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  /// Runs detection + segmentation on [imageFile].
  ///
  /// Pipeline:
  ///   1. Decode the capture and build a working image at 1024-longest-edge —
  ///      used for storage, display, the editor, and as YOLO's input frame.
  ///      Detection bbox / centroid / mask coords are reported in this space.
  ///   2. YOLO26 → list of `(bbox, centroid)` candidates on the working image.
  ///   3. Encode the working image **once** with the SAM encoder so every
  ///      detection's decoder pass shares the same global embedding (mirrors
  ///      what the instance editor does in manual mode). Then for each
  ///      candidate run the decoder with the YOLO centroid as a foreground
  ///      point prompt and the YOLO bbox as a box prompt; SAM reconstructs the
  ///      mask directly at working-image resolution, so no per-crop resampling
  ///      step is needed.
  ///   4. Encode working PNG + a coloured overlay for legacy callers.
  ///
  /// [onProgress] receives `(completed_segmentations, total)` as each
  /// SAM decode finishes — the background detection service uses it to drive
  /// the progress UI on the samples list.
  Future<WheatHeadResult> run(
    File imageFile, {
    void Function(int done, int total)? onProgress,
  }) async {
    final stopwatch = Stopwatch()..start();
    await _ensureWheatLoaded();

    final raw = await imageFile.readAsBytes();
    final fullRes = img.decodeImage(raw);
    if (fullRes == null) {
      throw StateError('Could not decode image: ${imageFile.path}');
    }

    final working = resizeLongestEdge(fullRes, _kSamInputSize);
    final w = working.width;
    final h = working.height;

    final centroids = await _detectCentroids(working);
    debugPrint('[wheat] yolo found ${centroids.length} candidates');
    onProgress?.call(0, centroids.length);

    final embedding = await prepareEditFromImage(working);
    final detections = <WheatHeadDetection>[];
    try {
      for (int i = 0; i < centroids.length; i++) {
        final cand = centroids[i];
        final mask = await predict(
          embedding: embedding,
          origW: w,
          origH: h,
          points: [cand.centroid],
          pointLabels: [1],
          bbox: cand.bbox,
        );
        detections.add(WheatHeadDetection(
          bbox: cand.bbox,
          centroid: cand.centroid,
          score: cand.score,
          mask: mask,
        ));
        onProgress?.call(i + 1, centroids.length);
      }
    } finally {
      embedding.release();
    }

    final imagePng = Uint8List.fromList(img.encodePng(working));
    final overlayPng = renderInstanceOverlay(detections, w, h);
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

  // ---------- Public editing API ----------

  /// Loads [imageFile] and produces the working-resolution image used by SAM.
  /// Caller owns the returned bytes; same coordinate space as everything in
  /// [WheatHeadResult].
  Future<({Uint8List png, int width, int height})> decodeWorking(
    File imageFile,
  ) async {
    final raw = await imageFile.readAsBytes();
    final decoded = img.decodeImage(raw);
    if (decoded == null) {
      throw StateError('Could not decode image: ${imageFile.path}');
    }
    final working = resizeLongestEdge(decoded, _kSamInputSize);
    return (
      png: Uint8List.fromList(img.encodePng(working)),
      width: working.width,
      height: working.height,
    );
  }

  /// Builds a SAM embedding from working-image PNG bytes (e.g. the bytes
  /// stored in the database) so the instance editor doesn't need to re-decode
  /// the original capture every time.
  Future<EditableEmbedding> prepareEditFromPng(Uint8List workingPng) async {
    final decoded = img.decodeImage(workingPng);
    if (decoded == null) {
      throw StateError('Could not decode working PNG');
    }
    return prepareEditFromImage(decoded);
  }

  /// Same as [prepareEditFromPng] but accepts an already-decoded image. Used
  /// by the per-frame pipelines, which hold the working image as `img.Image`
  /// and would otherwise round-trip through PNG just to feed the encoder.
  Future<EditableEmbedding> prepareEditFromImage(img.Image working) async {
    await ensureSam();
    final embedding = await _runEncoder(working);
    return EditableEmbedding._(embedding);
  }

  /// Re-runs the SAM decoder with arbitrary prompts. Coordinates in [points]
  /// and [bbox] are in working-image space (same units as
  /// [WheatHeadDetection.bbox]).
  ///
  /// [pointLabels] uses SAM's convention: 1 = foreground, 0 = background.
  Future<Uint8List> predict({
    required EditableEmbedding embedding,
    required int origW,
    required int origH,
    List<Offset> points = const [],
    List<int> pointLabels = const [],
    Rect? bbox,
  }) async {
    if (points.length != pointLabels.length) {
      throw ArgumentError('points and pointLabels length mismatch');
    }
    if (points.isEmpty && bbox == null) {
      throw ArgumentError('predict() needs at least one prompt');
    }
    return _runDecoderRaw(
      embedding: embedding._embedding,
      points: points,
      labels: pointLabels,
      bbox: bbox,
      origH: origH,
      origW: origW,
    );
  }

  /// Encode a binary mask (1 = inside) sized [w]×[h] as a single-channel
  /// PNG suitable for SQL storage.
  Uint8List encodeMaskPng(Uint8List mask, int w, int h) {
    final image = img.Image(width: w, height: h, numChannels: 1);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        image.setPixelR(x, y, mask[y * w + x] != 0 ? 255 : 0);
      }
    }
    return Uint8List.fromList(img.encodePng(image));
  }

  /// Inverse of [encodeMaskPng].
  ({Uint8List mask, int width, int height}) decodeMaskPng(Uint8List png) {
    final decoded = img.decodeImage(png);
    if (decoded == null) {
      throw StateError('Could not decode mask PNG');
    }
    final w = decoded.width;
    final h = decoded.height;
    final out = Uint8List(w * h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (decoded.getPixel(x, y).r > 127) out[y * w + x] = 1;
      }
    }
    return (mask: out, width: w, height: h);
  }

  /// Decodes the working image PNG once. Callers that need the raw RGB pixels
  /// (e.g. the FHB analyzer) reuse the returned object across instances
  /// instead of re-decoding for every spike.
  img.Image decodeWorkingImage(Uint8List workingPng) {
    final decoded = img.decodeImage(workingPng);
    if (decoded == null) {
      throw StateError('Could not decode working PNG');
    }
    return decoded;
  }

  /// Decode the original capture file once so the detection pipeline can
  /// render every per-instance preview tile from full-resolution pixels
  /// without re-reading the JPEG per detection.
  Future<img.Image> decodeImageFile(File file) async {
    final bytes = await file.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw StateError('Could not decode image file: ${file.path}');
    }
    return decoded;
  }

  /// Runs the wheat FHB disease-detection sub-pipeline on a decoded mask.
  /// Returns a [FhbReport] with per-pixel HSV classification + FHB ratio
  /// over the wheat-head spike mask.
  ///
  /// Async to keep parity with [GrapeLeafPipeline.analyzeDisease], whose
  /// SwinV2 inference is genuinely async; the wheat path itself is CPU-
  /// bound.
  Future<FhbReport> analyzeDisease({
    required img.Image workingImage,
    required Uint8List mask,
    required int maskWidth,
    required int maskHeight,
    required Rect bbox,
  }) async {
    return FhbAnalyzer.instance.analyze(
      workingImage: workingImage,
      spikeMask: mask,
      maskWidth: maskWidth,
      maskHeight: maskHeight,
      bbox: bbox,
    );
  }

  /// Renders a per-instance disease preview tile: cropped from [source] (so
  /// the tile is at the source image's pixel density — typically the full
  /// resolution capture) with each pixel tinted according to the
  /// classification map (green = healthy, red = necrotic, yellow = other).
  ///
  /// The classification map and `bbox` are in working-image (mask) coords;
  /// they're scaled up on the fly via `source.width / report.maskWidth` so
  /// callers don't have to upsample anything before invoking this.
  Uint8List renderDiseasePreview({
    required img.Image source,
    required FhbReport report,
    required Rect bbox,
    int padding = 16,
  }) {
    final crop = _computePaddedCrop(
      source: source,
      maskWidth: report.maskWidth,
      maskHeight: report.maskHeight,
      bbox: bbox,
      padding: padding,
    );
    final canvas = img.Image(
      width: crop.cropW,
      height: crop.cropH,
      numChannels: 4,
    );
    for (int y = 0; y < crop.cropH; y++) {
      for (int x = 0; x < crop.cropW; x++) {
        final src = source.getPixel(crop.sLeft + x, crop.sTop + y);
        canvas.setPixelRgba(
            x, y, src.r.toInt(), src.g.toInt(), src.b.toInt(), 255);
      }
    }
    _paintDiseaseOverlayScaled(
      canvas: canvas,
      sourceLeft: crop.sLeft,
      sourceTop: crop.sTop,
      sourceWidth: source.width,
      sourceHeight: source.height,
      classification: report.classification,
      maskWidth: report.maskWidth,
      maskHeight: report.maskHeight,
    );
    img.drawRect(
      canvas,
      x1: 0,
      y1: 0,
      x2: crop.cropW - 1,
      y2: crop.cropH - 1,
      color: img.ColorRgba8(255, 255, 255, 60),
    );
    return Uint8List.fromList(img.encodePng(canvas));
  }

  /// Single instance mask + its working-image dimensions. Used by
  /// [renderCombinedSegmentationOverlay] to build a stack of mask layers.
  /// The masks are expected to be in the same coordinate space as `width` /
  /// `height` (i.e. working-image resolution).
  ///
  /// Defined as a typedef rather than a class because Dart records can't be
  /// used as named parameters cleanly when they have to flow through `late`
  /// fields elsewhere.
  /// Combined per-instance segmentation overlay: one RGBA PNG sized to the
  /// working image, with each mask painted as a green tint and an outline
  /// drawn around its boundary. Pixels outside every mask are transparent so
  /// the viewer can stack this layer over the background image.
  ///
  /// Each entry in [masks] supplies a mask buffer (1 = inside, 0 = outside)
  /// at `width × height`. Entries with mismatched dimensions are skipped.
  Uint8List renderCombinedSegmentationOverlay({
    required int width,
    required int height,
    required List<({Uint8List mask, int width, int height})> masks,
  }) {
    final canvas = img.Image(width: width, height: height, numChannels: 4);
    img.fill(canvas, color: img.ColorRgba8(0, 0, 0, 0));
    final fill = img.ColorRgba8(0, 230, 118, 110);
    final outline = img.ColorRgba8(0, 230, 118, 230);
    for (final entry in masks) {
      if (entry.width != width || entry.height != height) continue;
      final m = entry.mask;
      for (int y = 0; y < height; y++) {
        final row = y * width;
        for (int x = 0; x < width; x++) {
          if (m[row + x] == 0) continue;
          // Outline = inside but at least one neighbour outside.
          final left = x > 0 && m[row + (x - 1)] != 0;
          final right = x < width - 1 && m[row + (x + 1)] != 0;
          final up = y > 0 && m[(y - 1) * width + x] != 0;
          final down = y < height - 1 && m[(y + 1) * width + x] != 0;
          if (!(left && right && up && down)) {
            canvas.setPixel(x, y, outline);
          } else {
            canvas.setPixel(x, y, fill);
          }
        }
      }
    }
    return Uint8List.fromList(img.encodePng(canvas));
  }

  /// Combines the per-instance classification maps into a single full-working-
  /// image RGBA overlay. Pixels outside every spike mask are transparent.
  /// Used by the sample viewer when the user toggles to "Disease" mode.
  Uint8List renderCombinedDiseaseOverlay({
    required int width,
    required int height,
    required List<FhbReport> reports,
  }) {
    final canvas = img.Image(width: width, height: height, numChannels: 4);
    img.fill(canvas, color: img.ColorRgba8(0, 0, 0, 0));
    for (final r in reports) {
      if (r.maskWidth != width || r.maskHeight != height) continue;
      _writeClassToCanvas(
        canvas: canvas,
        classification: r.classification,
        width: width,
        height: height,
      );
    }
    return Uint8List.fromList(img.encodePng(canvas));
  }

  void _writeClassToCanvas({
    required img.Image canvas,
    required Uint8List classification,
    required int width,
    required int height,
  }) {
    final greenFill = img.ColorRgba8(0, 200, 0, 130);
    final necroFill = img.ColorRgba8(220, 30, 30, 170);
    final otherFill = img.ColorRgba8(220, 200, 0, 130);
    for (int y = 0; y < height; y++) {
      final row = y * width;
      for (int x = 0; x < width; x++) {
        final c = classification[row + x];
        if (c == kClassGreen) {
          canvas.setPixel(x, y, greenFill);
        } else if (c == kClassNecrotic) {
          canvas.setPixel(x, y, necroFill);
        } else if (c == kClassOther) {
          canvas.setPixel(x, y, otherFill);
        }
        // c == kClassOutside → leave transparent.
      }
    }
  }

  /// Crops a region around [bbox] (working-image coords) from [source] with
  /// [padding] working-image pixels of slack on each side, then overlays the
  /// segmentation outline and a green tint. The returned PNG is at
  /// `source`'s pixel density — pass the original full-res capture as
  /// `source` for sharp preview tiles, or the working-image as a fallback.
  Uint8List renderInstancePreview({
    required img.Image source,
    required Uint8List mask,
    required int maskWidth,
    required int maskHeight,
    required Rect bbox,
    int padding = 16,
  }) {
    final crop = _computePaddedCrop(
      source: source,
      maskWidth: maskWidth,
      maskHeight: maskHeight,
      bbox: bbox,
      padding: padding,
    );
    final canvas = img.Image(
      width: crop.cropW,
      height: crop.cropH,
      numChannels: 4,
    );
    for (int y = 0; y < crop.cropH; y++) {
      for (int x = 0; x < crop.cropW; x++) {
        final src = source.getPixel(crop.sLeft + x, crop.sTop + y);
        canvas.setPixelRgba(
            x, y, src.r.toInt(), src.g.toInt(), src.b.toInt(), 255);
      }
    }
    _paintMaskOverlayScaled(
      canvas: canvas,
      sourceLeft: crop.sLeft,
      sourceTop: crop.sTop,
      sourceWidth: source.width,
      sourceHeight: source.height,
      mask: mask,
      maskWidth: maskWidth,
      maskHeight: maskHeight,
    );
    img.drawRect(
      canvas,
      x1: 0,
      y1: 0,
      x2: crop.cropW - 1,
      y2: crop.cropH - 1,
      color: img.ColorRgba8(255, 255, 255, 60),
    );
    return Uint8List.fromList(img.encodePng(canvas));
  }

  /// Compute the padded crop in source-image pixel coordinates given a
  /// working-coord [bbox] and [padding]. Mask coords are scaled by the
  /// source/mask ratio (assumed isotropic — both share the same aspect).
  _PaddedCrop _computePaddedCrop({
    required img.Image source,
    required int maskWidth,
    required int maskHeight,
    required Rect bbox,
    required int padding,
  }) {
    final sxToSrc = source.width / maskWidth;
    final syToSrc = source.height / maskHeight;
    final mLeft = (bbox.left - padding).floor().clamp(0, maskWidth - 1);
    final mTop = (bbox.top - padding).floor().clamp(0, maskHeight - 1);
    final mRight = (bbox.right + padding).ceil().clamp(mLeft + 1, maskWidth);
    final mBottom =
        (bbox.bottom + padding).ceil().clamp(mTop + 1, maskHeight);
    final sLeft = (mLeft * sxToSrc).floor().clamp(0, source.width - 1);
    final sTop = (mTop * syToSrc).floor().clamp(0, source.height - 1);
    final sRight = (mRight * sxToSrc).ceil().clamp(sLeft + 1, source.width);
    final sBottom =
        (mBottom * syToSrc).ceil().clamp(sTop + 1, source.height);
    return _PaddedCrop(
      sLeft: sLeft,
      sTop: sTop,
      cropW: sRight - sLeft,
      cropH: sBottom - sTop,
    );
  }

  /// Paint the segmentation tint + outline over [canvas] (which sits in
  /// source-image pixel space) by sampling [mask] at the corresponding
  /// working-image coords. The outline thickness scales with the
  /// source/mask ratio so it remains visible even on full-res tiles.
  void _paintMaskOverlayScaled({
    required img.Image canvas,
    required int sourceLeft,
    required int sourceTop,
    required int sourceWidth,
    required int sourceHeight,
    required Uint8List mask,
    required int maskWidth,
    required int maskHeight,
  }) {
    final invX = maskWidth / sourceWidth;
    final invY = maskHeight / sourceHeight;
    final outline = img.ColorRgba8(0, 230, 118, 255);
    bool insideAt(int sx, int sy) {
      final mx = (sx * invX).floor();
      final my = (sy * invY).floor();
      if (mx < 0 || my < 0 || mx >= maskWidth || my >= maskHeight) {
        return false;
      }
      return mask[my * maskWidth + mx] != 0;
    }

    // Outline thickness in source pixels = roughly one mask cell, so the
    // edge stays visible after the source-image upscale. Capped to avoid
    // overpowering tiny crops.
    final thicken = math.max(
      1,
      math.min(3, ((sourceWidth / maskWidth).round() + 1) ~/ 2),
    );

    for (int y = 0; y < canvas.height; y++) {
      final sy = sourceTop + y;
      for (int x = 0; x < canvas.width; x++) {
        final sx = sourceLeft + x;
        if (!insideAt(sx, sy)) continue;
        final px = canvas.getPixel(x, y);
        const a = 110 / 255.0;
        final r = (px.r * (1 - a) + 0 * a).round();
        final g = (px.g * (1 - a) + 230 * a).round();
        final b = (px.b * (1 - a) + 118 * a).round();
        canvas.setPixelRgba(x, y, r, g, b, 255);
        final isEdge = !insideAt(sx - thicken, sy) ||
            !insideAt(sx + thicken, sy) ||
            !insideAt(sx, sy - thicken) ||
            !insideAt(sx, sy + thicken);
        if (isEdge) canvas.setPixel(x, y, outline);
      }
    }
  }

  /// Like [_paintMaskOverlayScaled] but tints each pixel by its
  /// classification value (green / necrotic / other) instead of a single
  /// segmentation tint. No outline — the disease overlay's value is the
  /// per-pixel colour.
  void _paintDiseaseOverlayScaled({
    required img.Image canvas,
    required int sourceLeft,
    required int sourceTop,
    required int sourceWidth,
    required int sourceHeight,
    required Uint8List classification,
    required int maskWidth,
    required int maskHeight,
  }) {
    final invX = maskWidth / sourceWidth;
    final invY = maskHeight / sourceHeight;
    const alpha = 0.55;
    for (int y = 0; y < canvas.height; y++) {
      final sy = sourceTop + y;
      final my = (sy * invY).floor();
      if (my < 0 || my >= maskHeight) continue;
      for (int x = 0; x < canvas.width; x++) {
        final sx = sourceLeft + x;
        final mx = (sx * invX).floor();
        if (mx < 0 || mx >= maskWidth) continue;
        final c = classification[my * maskWidth + mx];
        late final int tr;
        late final int tg;
        late final int tb;
        switch (c) {
          case kClassGreen:
            tr = 0;
            tg = 200;
            tb = 0;
            break;
          case kClassNecrotic:
            tr = 220;
            tg = 30;
            tb = 30;
            break;
          case kClassOther:
            tr = 220;
            tg = 200;
            tb = 0;
            break;
          default:
            continue;
        }
        final px = canvas.getPixel(x, y);
        final r = (px.r * (1 - alpha) + tr * alpha).round();
        final g = (px.g * (1 - alpha) + tg * alpha).round();
        final b = (px.b * (1 - alpha) + tb * alpha).round();
        canvas.setPixelRgba(x, y, r, g, b, 255);
      }
    }
  }

  // ---------- YOLO26 ----------

  Future<List<Candidate>> _detectCentroids(img.Image working) async {
    final session = _yolo!;
    final letterboxed = letterbox(working, _kYoloInputSize);
    final input = imageToYoloTensor(letterboxed.image);
    final tensor = OrtValueTensor.createTensorWithDataList(
      input,
      [1, 3, _kYoloInputSize, _kYoloInputSize],
    );
    final yoloInputs = session.inputNames;
    if (yoloInputs.isEmpty) {
      tensor.release();
      throw StateError('YOLO ONNX session has no input names');
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

  /// YOLO26n-seg ONNX output (post-NMS, end-to-end): [1, 300, 38] where each
  /// row is `[x1, y1, x2, y2, conf, cls, ...32 mask coeffs]` in 640-letterbox
  /// space. We only consume the box + score columns and let SAM produce the
  /// per-instance mask. Coordinates are in (x1,y1,x2,y2).
  static const int _kYoloFeatures = 38;

  List<Candidate> _parseYoloOutput(
    Object? raw, {
    required LetterboxResult letterbox,
    required int origW,
    required int origH,
  }) {
    final flat = flattenToDoubles(raw);
    if (flat.isEmpty) return const [];
    if (flat.length % _kYoloFeatures != 0) {
      debugPrint('[wheat] unexpected yolo output length ${flat.length}');
      return const [];
    }
    final n = flat.length ~/ _kYoloFeatures;

    final candidates = <Candidate>[];
    for (int i = 0; i < n; i++) {
      final off = i * _kYoloFeatures;
      final x1 = flat[off];
      final y1 = flat[off + 1];
      final x2 = flat[off + 2];
      final y2 = flat[off + 3];
      final score = flat[off + 4];
      if (score < _kScoreThreshold) continue;
      candidates.add(remap(
        cx: (x1 + x2) / 2,
        cy: (y1 + y2) / 2,
        w: (x2 - x1).abs(),
        h: (y2 - y1).abs(),
        score: score,
        letterbox: letterbox,
        origW: origW,
        origH: origH,
        xyxy: true,
      ));
    }

    candidates.sort((a, b) => b.score.compareTo(a.score));
    final kept = nms(candidates, _kNmsIou);
    if (kept.length > _kMaxDetections) {
      return kept.sublist(0, _kMaxDetections);
    }
    return kept;
  }

  /// Recursively walk whatever structure `OrtValue.value` returned (nested
  /// `List`s, `Float32List`s, scalars) and produce a single flat
  /// `Float32List` of all leaf numbers. This is robust to the various
  /// representations the binding emits across versions/dtypes.
  Float32List flattenToDoubles(Object? raw) {
    final buf = <double>[];
    void walk(Object? v) {
      if (v == null) return;
      if (v is num) {
        buf.add(v.toDouble());
        return;
      }
      if (v is Float32List || v is Float64List ||
          v is Int32List || v is Int64List) {
        for (final n in (v as List)) {
          buf.add((n as num).toDouble());
        }
        return;
      }
      if (v is Iterable) {
        for (final e in v) {
          walk(e);
        }
        return;
      }
    }
    walk(raw);
    return Float32List.fromList(buf);
  }

  Candidate remap({
    required double cx,
    required double cy,
    required double w,
    required double h,
    required double score,
    required LetterboxResult letterbox,
    required int origW,
    required int origH,
    required bool xyxy,
  }) {
    // YOLO output is in 640-letterbox space. Undo letterbox to original.
    final scale = letterbox.scale;
    final padX = letterbox.padX;
    final padY = letterbox.padY;
    final ox = (cx - padX) / scale;
    final oy = (cy - padY) / scale;
    final ow = w / scale;
    final oh = h / scale;
    final left = (ox - ow / 2).clamp(0.0, origW.toDouble());
    final top = (oy - oh / 2).clamp(0.0, origH.toDouble());
    final right = (ox + ow / 2).clamp(0.0, origW.toDouble());
    final bottom = (oy + oh / 2).clamp(0.0, origH.toDouble());
    return Candidate(
      bbox: Rect.fromLTRB(left, top, right, bottom),
      centroid: Offset(ox.clamp(0, origW.toDouble()),
          oy.clamp(0, origH.toDouble())),
      score: score,
    );
  }

  List<Candidate> nms(List<Candidate> cands, double iouThresh) {
    final kept = <Candidate>[];
    final removed = List<bool>.filled(cands.length, false);
    for (int i = 0; i < cands.length; i++) {
      if (removed[i]) continue;
      kept.add(cands[i]);
      for (int j = i + 1; j < cands.length; j++) {
        if (removed[j]) continue;
        if (iou(cands[i].bbox, cands[j].bbox) > iouThresh) {
          removed[j] = true;
        }
      }
    }
    return kept;
  }

  double iou(Rect a, Rect b) {
    final ix = math.max(0.0, math.min(a.right, b.right) - math.max(a.left, b.left));
    final iy = math.max(0.0, math.min(a.bottom, b.bottom) - math.max(a.top, b.top));
    final inter = ix * iy;
    final union = a.width * a.height + b.width * b.height - inter;
    if (union <= 0) return 0;
    return inter / union;
  }

  // ---------- SAM-efficient encoder ----------

  Future<_Embedding> _runEncoder(img.Image working) async {
    final session = _samEncoder!;
    // SAM's decoder reconstructs masks by:
    //   1. computing `(H_resized, W_resized)` from `orig_im_size` such that
    //      the longest edge maps to the model input size (1024);
    //   2. cropping the 1024×1024 logits to `[0:H_resized, 0:W_resized]`;
    //   3. resizing that crop back to `orig_im_size`.
    // For step 2 to land on actual content, the encoder input MUST have the
    // image scaled so its longest edge == 1024 and pasted at (0, 0). Just
    // padding a small image (e.g. a 256-px crop) into the corner without
    // scaling leaves SAM looking at zero pixels for most of `[0:1024]` and
    // the mask we get back covers a region 4× larger than the actual content.
    final letterboxed = _letterboxTopLeft(working, _kSamInputSize);
    final input = _imageToSamTensor(letterboxed.image);
    final tensor = OrtValueTensor.createTensorWithDataList(
      input,
      [1, 3, _kSamInputSize, _kSamInputSize],
    );
    final encInputs = session.inputNames;
    if (encInputs.isEmpty) {
      tensor.release();
      throw StateError('SAM encoder session has no input names');
    }
    final inputName = encInputs.first;
    final runOpts = OrtRunOptions();
    final outputs = await session.runAsync(runOpts, {inputName: tensor});
    tensor.release();
    runOpts.release();
    if (outputs == null || outputs.isEmpty || outputs.first == null) {
      throw StateError('SAM encoder returned no output');
    }
    final value = outputs.first!;
    return _Embedding(
      value: value,
      letterbox: letterboxed,
    );
  }

  // ---------- SAM-efficient decoder ----------

  /// Generic decoder runner that accepts any combination of foreground/
  /// background point prompts plus an optional bbox. SAM's decoder expects
  /// at least one prompt; if both [points] and [bbox] are empty this throws.
  Future<Uint8List> _runDecoderRaw({
    required _Embedding embedding,
    List<Offset> points = const [],
    List<int> labels = const [],
    Rect? bbox,
    required int origH,
    required int origW,
  }) async {
    if (points.isEmpty && bbox == null) {
      throw ArgumentError('SAM decoder needs at least one prompt');
    }
    final session = _samDecoder!;
    final lb = embedding.letterbox;

    final coords = <double>[];
    final lbls = <double>[];
    for (int i = 0; i < points.length; i++) {
      coords.add(points[i].dx * lb.scale + lb.padX);
      coords.add(points[i].dy * lb.scale + lb.padY);
      lbls.add(labels[i].toDouble());
    }
    if (bbox != null) {
      coords.add(bbox.left * lb.scale + lb.padX);
      coords.add(bbox.top * lb.scale + lb.padY);
      coords.add(bbox.right * lb.scale + lb.padX);
      coords.add(bbox.bottom * lb.scale + lb.padY);
      lbls.add(2.0);
      lbls.add(3.0);
    }
    final n = lbls.length;

    final pointCoords = Float32List.fromList(coords);
    final pointLabels = Float32List.fromList(lbls);
    final maskInput = Float32List(1 * 1 * 256 * 256);
    final hasMask = Float32List.fromList([0.0]);
    final origSize = Float32List.fromList([origH.toDouble(), origW.toDouble()]);

    final coordsTensor = OrtValueTensor.createTensorWithDataList(
      pointCoords,
      [1, n, 2],
    );
    final labelsTensor = OrtValueTensor.createTensorWithDataList(
      pointLabels,
      [1, n],
    );
    final maskTensor = OrtValueTensor.createTensorWithDataList(
      maskInput,
      [1, 1, 256, 256],
    );
    final hasMaskTensor =
        OrtValueTensor.createTensorWithDataList(hasMask, [1]);
    final origSizeTensor =
        OrtValueTensor.createTensorWithDataList(origSize, [2]);

    final inputs = <String, OrtValue>{
      'image_embeddings': embedding.value,
      'point_coords': coordsTensor,
      'point_labels': labelsTensor,
      'mask_input': maskTensor,
      'has_mask_input': hasMaskTensor,
      'orig_im_size': origSizeTensor,
    };

    final runOpts = OrtRunOptions();
    final outputs = await session.runAsync(runOpts, inputs);
    runOpts.release();
    coordsTensor.release();
    labelsTensor.release();
    maskTensor.release();
    hasMaskTensor.release();
    origSizeTensor.release();

    if (outputs == null || outputs.isEmpty || outputs.first == null) {
      throw StateError('SAM decoder returned no output');
    }
    final masksValue = outputs.first!.value;
    for (final o in outputs) {
      o?.release();
    }
    return _binarizeMask(masksValue, origW: origW, origH: origH);
  }

  /// SAM mask logits → packed binary [H*W] Uint8List (1 = inside).
  ///
  /// We told SAM `orig_im_size = [origH, origW]`, so the decoder gives us
  /// `H * W` mask values regardless of how the data is nested. Flatten
  /// the structure and threshold at >0 (SAM's foreground convention).
  Uint8List _binarizeMask(Object? raw, {required int origW, required int origH}) {
    final flat = flattenToDoubles(raw);
    final expected = origW * origH;
    final out = Uint8List(expected);
    if (flat.length == expected) {
      for (int i = 0; i < expected; i++) {
        if (flat[i] > 0) out[i] = 1;
      }
      return out;
    }
    // Output came back at a different resolution than we asked for —
    // assume square-ish [H', W'] = sqrt(len), then nearest-resize.
    final side = math.sqrt(flat.length).round();
    if (side * side == flat.length) {
      final tmp = Uint8List(flat.length);
      for (int i = 0; i < flat.length; i++) {
        if (flat[i] > 0) tmp[i] = 1;
      }
      return _resizeNearest(tmp, side, side, origW, origH);
    }
    debugPrint('[wheat] SAM mask length ${flat.length} != $expected');
    return out;
  }

  Uint8List _resizeNearest(
    Uint8List src,
    int sw,
    int sh,
    int dw,
    int dh,
  ) {
    final dst = Uint8List(dw * dh);
    for (int y = 0; y < dh; y++) {
      final sy = (y * sh) ~/ dh;
      for (int x = 0; x < dw; x++) {
        final sx = (x * sw) ~/ dw;
        dst[y * dw + x] = src[sy * sw + sx];
      }
    }
    return dst;
  }

  // ---------- Image utilities ----------

  img.Image resizeLongestEdge(img.Image src, int target) {
    // Always scale the working image so its longest edge equals `target`
    // (down or up). SAM's decoder math assumes the encoder's input is
    // exactly `target` pixels on the long side, with the original image
    // pasted at (0,0). Skipping the upscale on small inputs leaves the
    // image at native scale inside a 1024×1024 canvas and breaks SAM's
    // internal `[0:h_resized, 0:w_resized]` crop.
    final long = math.max(src.width, src.height);
    if (long == target) return src;
    final scale = target / long;
    final w = (src.width * scale).round();
    final h = (src.height * scale).round();
    return img.copyResize(src, width: w, height: h, interpolation: img.Interpolation.linear);
  }

  /// Scale `src` so its longest edge equals `target`, paste at the top-left
  /// of a `target × target` zero-padded canvas, and record the scale used so
  /// the decoder can map prompt coords (which are in `src` coords) into the
  /// canvas's coord space.
  ///
  /// Used by the SAM encoder. Replaces a previous `_padTopLeft` that placed
  /// `src` without scaling — that worked while every input was already at
  /// 1024 longest edge, but breaks for the per-crop pipeline that hands the
  /// encoder a 256-px crop. SAM's mask reconstruction maps a `[0:1024]`
  /// region back to `orig_im_size`; if the actual content only spans
  /// `[0:256]`, the returned mask covers 4× the wrong area.
  LetterboxResult _letterboxTopLeft(img.Image src, int target) {
    final long = math.max(src.width, src.height);
    final scale = target / long;
    final newW = (src.width * scale).round();
    final newH = (src.height * scale).round();
    final resized = (newW == src.width && newH == src.height)
        ? src
        : img.copyResize(
            src,
            width: newW,
            height: newH,
            interpolation: img.Interpolation.linear,
          );
    final canvas = img.Image(width: target, height: target, numChannels: 3);
    img.fill(canvas, color: img.ColorRgb8(0, 0, 0));
    img.compositeImage(canvas, resized, dstX: 0, dstY: 0);
    return LetterboxResult(
      image: canvas,
      scale: scale,
      padX: 0,
      padY: 0,
    );
  }

  LetterboxResult letterbox(img.Image src, int target) {
    final scale = math.min(target / src.width, target / src.height);
    final newW = (src.width * scale).round();
    final newH = (src.height * scale).round();
    final resized = img.copyResize(src,
        width: newW, height: newH, interpolation: img.Interpolation.linear);
    final canvas = img.Image(width: target, height: target, numChannels: 3);
    img.fill(canvas, color: img.ColorRgb8(0, 0, 0));
    final padX = ((target - newW) / 2).floor();
    final padY = ((target - newH) / 2).floor();
    img.compositeImage(canvas, resized, dstX: padX, dstY: padY);
    return LetterboxResult(
      image: canvas,
      scale: scale,
      padX: padX.toDouble(),
      padY: padY.toDouble(),
    );
  }

  Float32List imageToYoloTensor(img.Image src) {
    final size = src.width;
    final out = Float32List(3 * size * size);
    final plane = size * size;
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final px = src.getPixel(x, y);
        final i = y * size + x;
        out[i] = px.r / 255.0;
        out[plane + i] = px.g / 255.0;
        out[2 * plane + i] = px.b / 255.0;
      }
    }
    return out;
  }

  Float32List _imageToSamTensor(img.Image src) {
    final size = src.width;
    final out = Float32List(3 * size * size);
    final plane = size * size;
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final px = src.getPixel(x, y);
        final i = y * size + x;
        out[i] = ((px.r / 255.0) - _kImagenetMean[0]) / _kImagenetStd[0];
        out[plane + i] = ((px.g / 255.0) - _kImagenetMean[1]) / _kImagenetStd[1];
        out[2 * plane + i] = ((px.b / 255.0) - _kImagenetMean[2]) / _kImagenetStd[2];
      }
    }
    return out;
  }

  Uint8List renderInstanceOverlay(
    List<WheatHeadDetection> detections,
    int w,
    int h,
  ) {
    final overlay = img.Image(width: w, height: h, numChannels: 4);
    img.fill(overlay, color: img.ColorRgba8(0, 0, 0, 0));
    final fill = img.ColorRgba8(0, 230, 118, 110);
    final outline = img.ColorRgba8(0, 230, 118, 230);
    final dot = img.ColorRgba8(255, 64, 129, 255);
    for (final d in detections) {
      final mask = d.mask;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          if (mask[y * w + x] != 0) {
            overlay.setPixel(x, y, fill);
          }
        }
      }
      img.drawRect(
        overlay,
        x1: d.bbox.left.round(),
        y1: d.bbox.top.round(),
        x2: d.bbox.right.round(),
        y2: d.bbox.bottom.round(),
        color: outline,
        thickness: 2,
      );
      img.fillCircle(
        overlay,
        x: d.centroid.dx.round(),
        y: d.centroid.dy.round(),
        radius: 4,
        color: dot,
      );
    }
    return Uint8List.fromList(img.encodePng(overlay));
  }
}

class Candidate {
  const Candidate({
    required this.bbox,
    required this.centroid,
    required this.score,
  });
  final Rect bbox;
  final Offset centroid;
  final double score;
}

class LetterboxResult {
  const LetterboxResult({
    required this.image,
    required this.scale,
    required this.padX,
    required this.padY,
  });
  final img.Image image;
  final double scale;
  final double padX;
  final double padY;
}

class _Embedding {
  _Embedding({required this.value, required this.letterbox});
  final OrtValue value;
  final LetterboxResult letterbox;
  void release() => value.release();
}

class _PaddedCrop {
  const _PaddedCrop({
    required this.sLeft,
    required this.sTop,
    required this.cropW,
    required this.cropH,
  });
  final int sLeft;
  final int sTop;
  final int cropW;
  final int cropH;
}
