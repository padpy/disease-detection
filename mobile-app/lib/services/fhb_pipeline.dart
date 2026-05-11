import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:gopher_eye/services/disease_morphology.dart';
import 'package:image/image.dart' as img;

export 'package:gopher_eye/services/disease_morphology.dart'
    show kClassOutside, kClassOther, kClassGreen, kClassNecrotic;

/// HSV bands used by the notebook's per-pixel classifier (OpenCV convention:
/// H ∈ [0, 180], S/V ∈ [0, 255]).
class FhbThresholds {
  const FhbThresholds({
    this.greenHueMin = 30,
    this.greenHueMax = 90,
    this.greenSatMin = 5,
    this.greenValMin = 5,
    this.necroHueMin = 0,
    this.necroHueMax = 30,
    this.necroSatMin = 0,
    this.necroValMin = 0,
  });

  final int greenHueMin;
  final int greenHueMax;
  final int greenSatMin;
  final int greenValMin;
  final int necroHueMin;
  final int necroHueMax;
  final int necroSatMin;
  final int necroValMin;

  static const FhbThresholds defaults = FhbThresholds();
}

/// Output of a disease analyzer. The classification map is sized
/// [maskWidth] × [maskHeight] (same as the working image / instance mask),
/// with values from [kClassOther] / [kClassGreen] / [kClassNecrotic] inside
/// the instance mask and 0 outside.
///
/// Originally written for FHB on wheat heads — the field semantics generalise
/// to other modes (e.g. grape leaf disease), with `green` standing for healthy
/// pixels and `necrotic` for diseased pixels regardless of the underlying
/// taxonomy. The class name and DB column prefix (`fhb_*`) are kept for
/// backward compatibility.
class FhbReport {
  const FhbReport({
    required this.greenCount,
    required this.necroticCount,
    required this.otherCount,
    required this.totalPixels,
    required this.fhbRatio,
    required this.severity,
    required this.classification,
    required this.maskWidth,
    required this.maskHeight,
  });

  final int greenCount;
  final int necroticCount;
  final int otherCount;
  final int totalPixels;

  /// Disease ratio expressed as `necrotic / (necrotic + green)` — matches the
  /// chart in the notebook. 0 when there are no green/necrotic pixels.
  final double fhbRatio;
  final String severity;

  /// Packed per-pixel class map (one byte per pixel, row-major).
  final Uint8List classification;
  final int maskWidth;
  final int maskHeight;
}

/// Runs the FHB disease-detection sub-pipeline (HSV classification → healthy
/// closure → small-contour filter) used by the FHB notebook on a single
/// segmentation instance.
class FhbAnalyzer {
  const FhbAnalyzer();

  static const FhbAnalyzer instance = FhbAnalyzer();

  /// Closure structuring element (notebook uses an elliptical 5×5).
  static const int _kClosureRadius = 2; // 5×5 ellipse

  /// FHB connected components below `total_mask_pixels * _kMinNecroFraction`
  /// are reclassified as healthy. The user spec asked for 1/32.
  static const double _kMinNecroFraction = 1 / 32;

  /// Closure radius applied to the "other" mask before dropping tiny bodies.
  static const int _kOtherClosureRadius = 3;

  /// Connected components of "other" pixels below this size are removed
  /// from the spike mask after the closure consolidates nearby specks.
  static const int _kOtherMinArea = 10;

  FhbReport analyze({
    required img.Image workingImage,
    required Uint8List spikeMask,
    required int maskWidth,
    required int maskHeight,
    required Rect bbox,
    FhbThresholds thresholds = FhbThresholds.defaults,
  }) {
    if (workingImage.width != maskWidth || workingImage.height != maskHeight) {
      throw ArgumentError(
        'workingImage size (${workingImage.width}×${workingImage.height}) '
        'must equal mask size ($maskWidth×$maskHeight)',
      );
    }

    final classification = Uint8List(maskWidth * maskHeight);
    int totalPixels = 0;
    int greenCount = 0;
    int necroticCount = 0;

    final rgb = workingImage.getBytes(order: img.ChannelOrder.rgb);
    const stride = 3;

    final x0 = bbox.left.floor().clamp(0, maskWidth - 1);
    final y0 = bbox.top.floor().clamp(0, maskHeight - 1);
    final x1 = bbox.right.ceil().clamp(x0 + 1, maskWidth);
    final y1 = bbox.bottom.ceil().clamp(y0 + 1, maskHeight);

    for (int y = y0; y < y1; y++) {
      final row = y * maskWidth;
      for (int x = x0; x < x1; x++) {
        final i = row + x;
        if (spikeMask[i] == 0) continue;
        totalPixels++;
        final p = i * stride;
        final r = rgb[p];
        final g = rgb[p + 1];
        final b = rgb[p + 2];
        final hsv = rgbToHsvOpenCv(r, g, b);
        final h = hsv[0];
        final s = hsv[1];
        final v = hsv[2];
        if (h >= thresholds.greenHueMin &&
            h <= thresholds.greenHueMax &&
            s >= thresholds.greenSatMin &&
            v >= thresholds.greenValMin) {
          classification[i] = kClassGreen;
          greenCount++;
        } else if (h >= thresholds.necroHueMin &&
            h <= thresholds.necroHueMax &&
            s >= thresholds.necroSatMin &&
            v >= thresholds.necroValMin) {
          classification[i] = kClassNecrotic;
          necroticCount++;
        } else {
          classification[i] = kClassOther;
        }
      }
    }

    if (totalPixels == 0) {
      return FhbReport(
        greenCount: 0,
        necroticCount: 0,
        otherCount: 0,
        totalPixels: 0,
        fhbRatio: 0,
        severity: 'Healthy',
        classification: classification,
        maskWidth: maskWidth,
        maskHeight: maskHeight,
      );
    }

    // ---------- Stage 2: healthy (green) morphological closure ----------
    morphologicalClosureOfClass(
      classification: classification,
      instanceMask: spikeMask,
      width: maskWidth,
      height: maskHeight,
      x0: x0,
      y0: y0,
      x1: x1,
      y1: y1,
      radius: _kClosureRadius,
      targetClass: kClassGreen,
      protectedClass: kClassOther,
    );

    // ---------- Stage 3: drop tiny FHB connected components ----------
    // Threshold is 1/32 of (healthy + unhealthy) pixels — i.e. of the
    // classified area, excluding 'other'. Closure only converts necrotic↔green
    // so the pre-closure sum is identical to the post-closure sum, making the
    // pre-closure counts safe to reuse here.
    final minArea =
        ((greenCount + necroticCount) * _kMinNecroFraction).floor();
    filterSmallContoursOfClass(
      classification: classification,
      instanceMask: spikeMask,
      width: maskWidth,
      height: maskHeight,
      x0: x0,
      y0: y0,
      x1: x1,
      y1: y1,
      minArea: minArea,
      targetClass: kClassNecrotic,
      replacementClass: kClassGreen,
    );

    // ---------- Stage 4: drop tiny "other" bodies from the mask ----------
    // Detect "other" pixels, run a 3-px closure to consolidate nearby
    // specks, then remove connected components smaller than 10 px by
    // reclassifying them to `outside` so they fall out of the recount.
    removeSmallOtherComponents(
      classification: classification,
      instanceMask: spikeMask,
      width: maskWidth,
      height: maskHeight,
      x0: x0,
      y0: y0,
      x1: x1,
      y1: y1,
      closureRadius: _kOtherClosureRadius,
      minArea: _kOtherMinArea,
    );

    // Recount after morphology — counts above are pre-cleanup, and Stage 4
    // may have removed pixels from the spike entirely (kClassOutside).
    totalPixels = 0;
    greenCount = 0;
    necroticCount = 0;
    int otherCount = 0;
    for (int y = y0; y < y1; y++) {
      final row = y * maskWidth;
      for (int x = x0; x < x1; x++) {
        final i = row + x;
        if (spikeMask[i] == 0) continue;
        final cls = classification[i];
        if (cls == kClassOutside) continue;
        totalPixels++;
        switch (cls) {
          case kClassGreen:
            greenCount++;
            break;
          case kClassNecrotic:
            necroticCount++;
            break;
          default:
            otherCount++;
        }
      }
    }

    final denom = greenCount + necroticCount;
    final fhbRatio = denom == 0 ? 0.0 : necroticCount / denom;
    final severity = _severityFor(fhbRatio);

    return FhbReport(
      greenCount: greenCount,
      necroticCount: necroticCount,
      otherCount: otherCount,
      totalPixels: totalPixels,
      fhbRatio: fhbRatio,
      severity: severity,
      classification: classification,
      maskWidth: maskWidth,
      maskHeight: maskHeight,
    );
  }

  /// Severity label that lines up with the notebook's NG-ratio bins, but
  /// applied to FHB% (necrotic / (green+necrotic)) since that's the value the
  /// chart reports per spike.
  String _severityFor(double fhbRatio) {
    if (fhbRatio < 0.05) return 'Healthy';
    if (fhbRatio < 0.25) return 'Mild FHB';
    if (fhbRatio < 0.50) return 'Moderate FHB';
    return 'Severe FHB';
  }
}
