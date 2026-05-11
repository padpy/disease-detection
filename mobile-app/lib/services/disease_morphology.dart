import 'dart:math' as math;
import 'dart:typed_data';

/// Class IDs in the per-pixel classification map. 0 is reserved for
/// "outside the instance mask" so the map can stand alone without also
/// carrying the mask around.
const int kClassOutside = 0;
const int kClassOther = 1;
const int kClassGreen = 2;
const int kClassNecrotic = 3;

/// Convert RGB (0–255) → HSV with H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255].
/// Matches OpenCV `COLOR_BGR2HSV`/`COLOR_RGB2HSV`, which is what the
/// notebook's thresholds were tuned against.
List<int> rgbToHsvOpenCv(int r, int g, int b) {
  final cMax = math.max(r, math.max(g, b));
  final cMin = math.min(r, math.min(g, b));
  final delta = cMax - cMin;
  int h = 0;
  if (delta != 0) {
    double hf;
    if (cMax == r) {
      hf = 60.0 * (((g - b) / delta) % 6);
    } else if (cMax == g) {
      hf = 60.0 * (((b - r) / delta) + 2);
    } else {
      hf = 60.0 * (((r - g) / delta) + 4);
    }
    if (hf < 0) hf += 360;
    h = (hf / 2).round();
    if (h >= 180) h = 179;
  }
  final s = cMax == 0 ? 0 : ((delta * 255) / cMax).round();
  final v = cMax;
  return [h, s, v];
}

/// Morphological closure (dilate then erode) of pixels classified as
/// [targetClass] inside the instance mask, restoring [protectedClass] pixels
/// after closure so the dilation can't overwrite them.
///
/// Used to fill in small holes in the healthy region of an instance.
void morphologicalClosureOfClass({
  required Uint8List classification,
  required Uint8List instanceMask,
  required int width,
  required int height,
  required int x0,
  required int y0,
  required int x1,
  required int y1,
  required int radius,
  required int targetClass,
  required int protectedClass,
}) {
  final w = x1 - x0;
  final h = y1 - y0;
  if (w <= 0 || h <= 0) return;

  final targetBin = Uint8List(w * h);
  final protectedFlags = Uint8List(w * h);
  for (int y = 0; y < h; y++) {
    final globalRow = (y + y0) * width;
    for (int x = 0; x < w; x++) {
      final gi = globalRow + (x + x0);
      if (instanceMask[gi] == 0) continue;
      final c = classification[gi];
      if (c == targetClass) targetBin[y * w + x] = 1;
      if (c == protectedClass) protectedFlags[y * w + x] = 1;
    }
  }

  final dilated = _dilateEllipse(targetBin, w, h, radius);
  final closed = _erodeEllipse(dilated, w, h, radius);

  for (int y = 0; y < h; y++) {
    final globalRow = (y + y0) * width;
    for (int x = 0; x < w; x++) {
      final gi = globalRow + (x + x0);
      if (instanceMask[gi] == 0) continue;
      if (protectedFlags[y * w + x] == 1) continue;
      if (closed[y * w + x] != 0) {
        classification[gi] = targetClass;
      }
    }
  }
}

Uint8List _dilateEllipse(Uint8List src, int w, int h, int radius) {
  final out = Uint8List(w * h);
  final r2 = radius * radius;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      bool any = false;
      for (int ky = -radius; ky <= radius && !any; ky++) {
        final yy = y + ky;
        if (yy < 0 || yy >= h) continue;
        for (int kx = -radius; kx <= radius; kx++) {
          if (kx * kx + ky * ky > r2) continue;
          final xx = x + kx;
          if (xx < 0 || xx >= w) continue;
          if (src[yy * w + xx] != 0) {
            any = true;
            break;
          }
        }
      }
      if (any) out[y * w + x] = 1;
    }
  }
  return out;
}

Uint8List _erodeEllipse(Uint8List src, int w, int h, int radius) {
  final out = Uint8List(w * h);
  final r2 = radius * radius;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (src[y * w + x] == 0) continue;
      bool all = true;
      for (int ky = -radius; ky <= radius && all; ky++) {
        final yy = y + ky;
        for (int kx = -radius; kx <= radius && all; kx++) {
          if (kx * kx + ky * ky > r2) continue;
          final xx = x + kx;
          if (xx < 0 || xx >= w || yy < 0 || yy >= h) {
            all = false;
            break;
          }
          if (src[yy * w + xx] == 0) all = false;
        }
      }
      if (all) out[y * w + x] = 1;
    }
  }
  return out;
}

/// Cleans up [kClassOther] pixels inside an instance: builds a binary mask
/// of "other" pixels, applies an elliptical closure of [closureRadius] to
/// consolidate nearby specks, then drops 8-connected components smaller
/// than [minArea] by reclassifying their original "other" pixels to
/// [kClassOutside] (i.e. removing them from the spike mask for downstream
/// counts and overlays). The closure is computed on a scratch buffer so
/// neighbouring [kClassGreen]/[kClassNecrotic] pixels are never overwritten.
void removeSmallOtherComponents({
  required Uint8List classification,
  required Uint8List instanceMask,
  required int width,
  required int height,
  required int x0,
  required int y0,
  required int x1,
  required int y1,
  required int closureRadius,
  required int minArea,
}) {
  final w = x1 - x0;
  final h = y1 - y0;
  if (w <= 0 || h <= 0) return;

  final otherBin = Uint8List(w * h);
  for (int y = 0; y < h; y++) {
    final globalRow = (y + y0) * width;
    for (int x = 0; x < w; x++) {
      final gi = globalRow + (x + x0);
      if (instanceMask[gi] == 0) continue;
      if (classification[gi] == kClassOther) otherBin[y * w + x] = 1;
    }
  }

  final closed = closureRadius > 0
      ? _erodeEllipse(
          _dilateEllipse(otherBin, w, h, closureRadius),
          w,
          h,
          closureRadius,
        )
      : otherBin;

  if (minArea <= 1) return;

  final visited = Uint8List(w * h);
  final stack = <int>[];

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final li = y * w + x;
      if (visited[li] != 0) continue;
      if (closed[li] == 0) {
        visited[li] = 1;
        continue;
      }
      stack.clear();
      stack.add(li);
      visited[li] = 1;
      final component = <int>[li];
      while (stack.isNotEmpty) {
        final cur = stack.removeLast();
        final cy = cur ~/ w;
        final cx = cur - cy * w;
        for (int dy = -1; dy <= 1; dy++) {
          final ny = cy + dy;
          if (ny < 0 || ny >= h) continue;
          for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            final nx = cx + dx;
            if (nx < 0 || nx >= w) continue;
            final nli = ny * w + nx;
            if (visited[nli] != 0) continue;
            visited[nli] = 1;
            if (closed[nli] == 0) continue;
            stack.add(nli);
            component.add(nli);
          }
        }
      }
      if (component.length < minArea) {
        for (final localIdx in component) {
          final cy = localIdx ~/ w;
          final cx = localIdx - cy * w;
          final gi = (cy + y0) * width + (cx + x0);
          if (instanceMask[gi] != 0 && classification[gi] == kClassOther) {
            classification[gi] = kClassOutside;
          }
        }
      }
    }
  }
}

/// Re-label connected components of [targetClass] smaller than [minArea] as
/// [replacementClass]. Used to drop salt-and-pepper diseased pixels that
/// pass the per-pixel HSV gate but aren't part of an actual lesion.
void filterSmallContoursOfClass({
  required Uint8List classification,
  required Uint8List instanceMask,
  required int width,
  required int height,
  required int x0,
  required int y0,
  required int x1,
  required int y1,
  required int minArea,
  required int targetClass,
  required int replacementClass,
}) {
  if (minArea <= 1) return;
  final w = x1 - x0;
  final h = y1 - y0;
  if (w <= 0 || h <= 0) return;

  final visited = Uint8List(w * h);
  final stack = <int>[];

  for (int y = 0; y < h; y++) {
    final globalRow = (y + y0) * width;
    for (int x = 0; x < w; x++) {
      final li = y * w + x;
      if (visited[li] != 0) continue;
      final gi = globalRow + (x + x0);
      if (instanceMask[gi] == 0 || classification[gi] != targetClass) {
        visited[li] = 1;
        continue;
      }
      stack.clear();
      stack.add(li);
      visited[li] = 1;
      final component = <int>[li];
      while (stack.isNotEmpty) {
        final cur = stack.removeLast();
        final cy = cur ~/ w;
        final cx = cur - cy * w;
        for (int dy = -1; dy <= 1; dy++) {
          final ny = cy + dy;
          if (ny < 0 || ny >= h) continue;
          for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            final nx = cx + dx;
            if (nx < 0 || nx >= w) continue;
            final nli = ny * w + nx;
            if (visited[nli] != 0) continue;
            final ngi = (ny + y0) * width + (nx + x0);
            if (instanceMask[ngi] == 0 ||
                classification[ngi] != targetClass) {
              visited[nli] = 1;
              continue;
            }
            visited[nli] = 1;
            stack.add(nli);
            component.add(nli);
          }
        }
      }
      if (component.length < minArea) {
        for (final localIdx in component) {
          final cy = localIdx ~/ w;
          final cx = localIdx - cy * w;
          classification[(cy + y0) * width + (cx + x0)] = replacementClass;
        }
      }
    }
  }
}
