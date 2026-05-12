import 'dart:async';
import 'dart:io';
import 'dart:ui' show Offset, Rect;

import 'package:flutter/foundation.dart';
import 'package:gopher_eye/model/detection_mode.dart';
import 'package:gopher_eye/model/sample_instance.dart';
import 'package:gopher_eye/services/api_client.dart';
import 'package:gopher_eye/services/app_settings.dart';
import 'package:gopher_eye/services/fhb_pipeline.dart';
import 'package:gopher_eye/services/grape_leaf_pipeline.dart';
import 'package:gopher_eye/services/sample_repository.dart';
import 'package:gopher_eye/services/sync_service.dart';
import 'package:gopher_eye/services/wheat_head_pipeline.dart';
import 'package:image/image.dart' as img;

/// Status of the detection pipeline for a single sample. Statuses are
/// produced by [DetectionService] and consumed by the UI to drive progress
/// indicators on the samples list and the sample viewer.
sealed class DetectionStatus {
  const DetectionStatus();
}

class DetectionIdle extends DetectionStatus {
  const DetectionIdle();
}

class DetectionRunning extends DetectionStatus {
  const DetectionRunning({required this.phase, this.progress});

  /// Short human-readable phase label (e.g. `Detecting`, `Segmenting 4/22`,
  /// `Analyzing disease`).
  final String phase;

  /// Optional `[0.0 .. 1.0]` progress for the current phase. Null when the
  /// phase doesn't have a meaningful fractional progress (e.g. YOLO).
  final double? progress;
}

class DetectionCompleted extends DetectionStatus {
  const DetectionCompleted({
    required this.detectionCount,
    required this.elapsed,
  });
  final int detectionCount;
  final Duration elapsed;
}

class DetectionFailed extends DetectionStatus {
  const DetectionFailed(this.error);
  final String error;
}

/// Process-wide detection job manager. Holds one [ValueNotifier] per sample
/// id so widgets can listen for progress without rebuilding the whole list,
/// and serialises pipeline runs so a backlog of saved captures doesn't
/// thrash the ONNX sessions.
class DetectionService extends ChangeNotifier {
  DetectionService._();
  static final DetectionService instance = DetectionService._();

  final Map<int, ValueNotifier<DetectionStatus>> _statuses = {};

  /// FIFO queue of pending samples. We process one at a time because the
  /// WheatHeadPipeline holds shared ONNX sessions and running concurrent
  /// jobs would interleave their tensor inputs.
  final List<_PendingJob> _queue = [];
  bool _draining = false;

  /// Listenable for the given sample. Always returns the same instance, so
  /// callers can `addListener` once and rely on it sticking around.
  ValueListenable<DetectionStatus> statusFor(int sampleId) =>
      _ensureNotifier(sampleId);

  ValueNotifier<DetectionStatus> _ensureNotifier(int sampleId) {
    return _statuses.putIfAbsent(
      sampleId,
      () => ValueNotifier<DetectionStatus>(const DetectionIdle()),
    );
  }

  /// Drop in-memory state for a sample. Call after the sample row is deleted
  /// so the notifier doesn't leak.
  void forget(int sampleId) {
    _statuses.remove(sampleId)?.dispose();
    _queue.removeWhere((j) => j.sampleId == sampleId);
  }

  /// Schedule a detection run. Returns immediately; the job runs in the
  /// background. If a job for [sampleId] is already queued or running, this
  /// is a no-op.
  void enqueue({
    required int sampleId,
    required String filePath,
    required DetectionMode mode,
  }) {
    final notifier = _ensureNotifier(sampleId);
    final running = notifier.value is DetectionRunning;
    final queued = _queue.any((j) => j.sampleId == sampleId);
    if (running || queued) return;
    _queue.add(_PendingJob(
      sampleId: sampleId,
      filePath: filePath,
      mode: mode,
    ));
    notifier.value =
        const DetectionRunning(phase: 'Queued', progress: null);
    notifyListeners();
    unawaited(_drain());
  }

  /// Re-run detection for a sample whose pipeline already completed (e.g.
  /// the user tapped "Re-run detection" in the viewer). Equivalent to
  /// [enqueue] but tolerates the already-completed status.
  void requeue({
    required int sampleId,
    required String filePath,
    required DetectionMode mode,
  }) {
    final notifier = _ensureNotifier(sampleId);
    if (notifier.value is DetectionRunning) return;
    _queue.removeWhere((j) => j.sampleId == sampleId);
    _queue.add(_PendingJob(
      sampleId: sampleId,
      filePath: filePath,
      mode: mode,
    ));
    notifier.value =
        const DetectionRunning(phase: 'Queued', progress: null);
    notifyListeners();
    unawaited(_drain());
  }

  Future<void> _drain() async {
    if (_draining) return;
    _draining = true;
    try {
      while (_queue.isNotEmpty) {
        final job = _queue.removeAt(0);
        await _process(job);
      }
    } finally {
      _draining = false;
    }
  }

  Future<void> _process(_PendingJob job) async {
    final location = await AppSettings.getDetectionLocation();
    if (location == DetectionLocation.remote) {
      await _processRemoteDetect(job);
    } else {
      await _processAutoDetect(job);
    }
  }

  /// Remote pipeline: upload the capture to the server's existing
  /// ``/dl/segmentation*`` endpoint, poll until it reports complete, then
  /// hydrate the local instances list from ``/plant/data``. The server's
  /// returned masks are normalized polygons; we rasterize them at the
  /// original image resolution into binary mask PNGs so the editor + viewer
  /// behave the same as for locally-detected samples.
  Future<void> _processRemoteDetect(_PendingJob job) async {
    final notifier = _ensureNotifier(job.sampleId);
    notifier.value =
        const DetectionRunning(phase: 'Uploading', progress: null);
    final stopwatch = Stopwatch()..start();
    try {
      final file = File(job.filePath);
      if (!await file.exists()) {
        throw StateError('capture file missing: ${job.filePath}');
      }
      final bytes = await file.readAsBytes();
      final task = job.mode == DetectionMode.wheatFhb ? 'spike' : 'leaf';

      final plantId = await ApiClient.instance.submitDetection(
        imageBytes: bytes,
        filename: file.uri.pathSegments.last,
        task: task,
      );

      notifier.value =
          const DetectionRunning(phase: 'Server detecting', progress: null);
      const pollInterval = Duration(seconds: 2);
      const maxWait = Duration(minutes: 3);
      final deadline = DateTime.now().add(maxWait);
      String status = 'pending';
      while (status != 'complete') {
        if (DateTime.now().isAfter(deadline)) {
          throw TimeoutException(
            'Server detection timed out after ${maxWait.inMinutes}m',
            maxWait,
          );
        }
        await Future<void>.delayed(pollInterval);
        status = await ApiClient.instance.getPlantStatus(plantId);
      }

      notifier.value =
          const DetectionRunning(phase: 'Hydrating results', progress: null);
      final data = await ApiClient.instance.getPlantData(plantId);

      final pipe = WheatHeadPipeline.instance;
      final fullRes = await pipe.decodeImageFile(file);
      final imageW = fullRes.width;
      final imageH = fullRes.height;

      final boxes = (data['bounding_boxes'] as List?) ?? const [];
      final masks = (data['masks'] as List?) ?? const [];

      final repo = SampleRepository.instance;
      final now = DateTime.now();
      final toSave = <SampleInstance>[];

      for (var i = 0; i < boxes.length; i++) {
        final box = boxes[i];
        if (box is! List || box.length < 4) continue;
        final x1 = (box[0] as num).toDouble().clamp(0.0, 1.0) * imageW;
        final y1 = (box[1] as num).toDouble().clamp(0.0, 1.0) * imageH;
        final x2 = (box[2] as num).toDouble().clamp(0.0, 1.0) * imageW;
        final y2 = (box[3] as num).toDouble().clamp(0.0, 1.0) * imageH;
        final bbox = Rect.fromLTRB(x1, y1, x2, y2);
        final centroid = Offset(
          (bbox.left + bbox.right) / 2,
          (bbox.top + bbox.bottom) / 2,
        );

        final polygon = (i < masks.length && masks[i] is List)
            ? _polygonInPixels(masks[i] as List, imageW, imageH)
            : null;
        final maskBytes = _rasterizeMask(
          imageW,
          imageH,
          polygon: polygon,
          fallback: bbox,
        );
        final maskPng = pipe.encodeMaskPng(maskBytes, imageW, imageH);
        final previewPng = pipe.renderInstancePreview(
          source: fullRes,
          mask: maskBytes,
          maskWidth: imageW,
          maskHeight: imageH,
          bbox: bbox,
        );

        toSave.add(SampleInstance(
          sampleId: job.sampleId,
          idx: i,
          bbox: bbox,
          centroid: centroid,
          score: 1.0,
          imageWidth: imageW,
          imageHeight: imageH,
          maskPng: maskPng,
          previewPng: previewPng,
          createdAt: now,
          updatedAt: now,
        ));
      }

      // Persist a working image so downstream code (editor, sync) has the
      // same shape it gets in the local pipeline. Server returns
      // full-resolution masks, so the working image is just the original.
      final workingPng = Uint8List.fromList(img.encodePng(fullRes));
      await repo.saveWorkingImage(
        sampleId: job.sampleId,
        png: workingPng,
        width: imageW,
        height: imageH,
      );
      final saved =
          await repo.replaceInstances(sampleId: job.sampleId, instances: toSave);

      // Run the on-device disease analyzer over the server-supplied masks so
      // each instance gets a proper per-pixel classification map (and the
      // combined disease + segmentation overlays). The server only returns a
      // single ``FHB: 0.XX`` string per mask, which is fine as a numeric
      // ratio but doesn't give us the green/red/yellow tile the viewer
      // expects in "Disease" mode — without this pass the disease preview
      // falls back to the segmentation outline.
      notifier.value = DetectionRunning(
        phase: 'Analyzing disease',
        progress: saved.isEmpty ? null : 0,
      );
      await runDiseaseAnalysis(
        mode: job.mode,
        workingPng: workingPng,
        workingW: imageW,
        workingH: imageH,
        instances: saved,
        sampleId: job.sampleId,
        sourceOverride: fullRes,
        onProgress: (done, total) {
          notifier.value = DetectionRunning(
            phase: 'Analyzing disease $done/$total',
            progress: total > 0 ? done / total : null,
          );
        },
      );

      stopwatch.stop();
      notifier.value = DetectionCompleted(
        detectionCount: toSave.length,
        elapsed: stopwatch.elapsed,
      );
      notifyListeners();
      final hydrated = await repo.findById(job.sampleId);
      if (hydrated != null) {
        unawaited(SyncService.instance.pushSampleResults(hydrated));
      }
    } catch (e, st) {
      debugPrint('[remote-detect] sample ${job.sampleId} failed: $e\n$st');
      notifier.value = DetectionFailed('$e');
      notifyListeners();
    }
  }

  /// Convert YOLO's normalized polygon [(x, y), ...] (each in [0..1]) into
  /// pixel-space points for the original image.
  List<Offset> _polygonInPixels(List raw, int width, int height) {
    final out = <Offset>[];
    for (final pt in raw) {
      if (pt is! List || pt.length < 2) continue;
      final px = (pt[0] as num).toDouble() * width;
      final py = (pt[1] as num).toDouble() * height;
      out.add(Offset(px, py));
    }
    return out;
  }

  /// Rasterize [polygon] (or fall back to [fallback]'s bbox) into a binary
  /// mask. Output is a flat byte buffer with 1 inside, 0 outside, matching
  /// the convention used elsewhere in the pipeline.
  Uint8List _rasterizeMask(
    int width,
    int height, {
    List<Offset>? polygon,
    required Rect fallback,
  }) {
    final out = Uint8List(width * height);
    if (polygon != null && polygon.length >= 3) {
      _fillPolygon(out, width, height, polygon);
      return out;
    }
    final l = fallback.left.clamp(0, width - 1).toInt();
    final t = fallback.top.clamp(0, height - 1).toInt();
    final r = fallback.right.clamp(0, width - 1).toInt();
    final b = fallback.bottom.clamp(0, height - 1).toInt();
    for (var y = t; y <= b; y++) {
      final rowStart = y * width;
      for (var x = l; x <= r; x++) {
        out[rowStart + x] = 1;
      }
    }
    return out;
  }

  /// Scanline polygon fill — standard even-odd rule. Each scan-line collects
  /// edge intersections, sorts them, and fills between pairs.
  void _fillPolygon(
    Uint8List buf,
    int width,
    int height,
    List<Offset> poly,
  ) {
    if (poly.length < 3) return;
    int minY = height, maxY = 0;
    for (final p in poly) {
      final y = p.dy.toInt();
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (minY < 0) minY = 0;
    if (maxY >= height) maxY = height - 1;
    for (var y = minY; y <= maxY; y++) {
      final intersections = <double>[];
      for (var i = 0; i < poly.length; i++) {
        final a = poly[i];
        final b = poly[(i + 1) % poly.length];
        final ay = a.dy;
        final by = b.dy;
        if ((ay <= y && by > y) || (by <= y && ay > y)) {
          final t = (y - ay) / (by - ay);
          intersections.add(a.dx + t * (b.dx - a.dx));
        }
      }
      intersections.sort();
      for (var k = 0; k + 1 < intersections.length; k += 2) {
        var xs = intersections[k].toInt();
        var xe = intersections[k + 1].toInt();
        if (xs < 0) xs = 0;
        if (xe >= width) xe = width - 1;
        final rowStart = y * width;
        for (var x = xs; x <= xe; x++) {
          buf[rowStart + x] = 1;
        }
      }
    }
  }

  /// On-device pipeline: YOLO + SAM auto-detection, replace instances, then
  /// run per-instance disease analysis. Wheat dispatches to the FHB analyzer;
  /// grape dispatches to the SwinV2-backed analyzer in
  /// [WheatHeadPipeline.analyzeDisease].
  Future<void> _processAutoDetect(_PendingJob job) async {
    final notifier = _ensureNotifier(job.sampleId);
    notifier.value =
        const DetectionRunning(phase: 'Detecting', progress: null);
    final stopwatch = Stopwatch()..start();
    try {
      final pipe = WheatHeadPipeline.instance;
      Future<WheatHeadResult> runFor(File file) {
        switch (job.mode) {
          case DetectionMode.wheatFhb:
            return pipe.run(
              file,
              onProgress: (done, total) {
                notifier.value = DetectionRunning(
                  phase: total == 0
                      ? 'Segmenting'
                      : 'Segmenting $done/$total',
                  progress: total > 0 ? done / total : null,
                );
              },
            );
          case DetectionMode.grapeLeaf:
            return GrapeLeafPipeline.instance.run(
              file,
              onProgress: (done, total) {
                notifier.value = DetectionRunning(
                  phase: total == 0
                      ? 'Locating leaves'
                      : 'Locating leaves $done/$total',
                  progress: total > 0 ? done / total : null,
                );
              },
            );
        }
      }
      final result = await runFor(File(job.filePath));

      // Persist the working image (still needed by the editor + FHB) and
      // decode the original capture once so every per-instance preview
      // tile + the combined disease overlay can be rendered at full
      // resolution rather than from the downscaled working image.
      final repo = SampleRepository.instance;
      await repo.saveWorkingImage(
        sampleId: job.sampleId,
        png: result.imagePng,
        width: result.width,
        height: result.height,
      );
      final fullRes = await pipe.decodeImageFile(File(job.filePath));

      final now = DateTime.now();
      final toSave = <SampleInstance>[];
      final maskRefs = <({Uint8List mask, int width, int height})>[];
      for (int i = 0; i < result.detections.length; i++) {
        final det = result.detections[i];
        final maskPng =
            pipe.encodeMaskPng(det.mask, result.width, result.height);
        final previewPng = pipe.renderInstancePreview(
          source: fullRes,
          mask: det.mask,
          maskWidth: result.width,
          maskHeight: result.height,
          bbox: det.bbox,
        );
        maskRefs.add((
          mask: det.mask,
          width: result.width,
          height: result.height,
        ));
        toSave.add(SampleInstance(
          sampleId: job.sampleId,
          idx: i,
          bbox: det.bbox,
          centroid: det.centroid,
          score: det.score,
          imageWidth: result.width,
          imageHeight: result.height,
          maskPng: maskPng,
          previewPng: previewPng,
          createdAt: now,
          updatedAt: now,
        ));
      }
      final saved = await repo.replaceInstances(
        sampleId: job.sampleId,
        instances: toSave,
      );

      // Combined segmentation overlay covering every instance mask. Stored
      // at working-image resolution; the viewer stretches it to fit the
      // original image since both share the same aspect ratio.
      Uint8List? segPng;
      if (maskRefs.isNotEmpty) {
        segPng = pipe.renderCombinedSegmentationOverlay(
          width: result.width,
          height: result.height,
          masks: maskRefs,
        );
      }
      await repo.saveSegmentationOverlay(
        sampleId: job.sampleId,
        png: segPng,
      );

      notifier.value = DetectionRunning(
        phase: 'Analyzing disease',
        progress: saved.isEmpty ? null : 0,
      );
      await runDiseaseAnalysis(
        mode: job.mode,
        workingPng: result.imagePng,
        workingW: result.width,
        workingH: result.height,
        instances: saved,
        sampleId: job.sampleId,
        sourceOverride: fullRes,
        onProgress: (done, total) {
          notifier.value = DetectionRunning(
            phase: 'Analyzing disease $done/$total',
            progress: total > 0 ? done / total : null,
          );
        },
      );

      stopwatch.stop();
      notifier.value = DetectionCompleted(
        detectionCount: result.detections.length,
        elapsed: stopwatch.elapsed,
      );
      notifyListeners();
      // Best-effort backup once the heavy work is done. SyncService already
      // exits early when the user hasn't enabled sync, so this is a cheap
      // fire-and-forget for the common local-only case.
      final hydrated = await SampleRepository.instance.findById(job.sampleId);
      if (hydrated != null) {
        unawaited(SyncService.instance.pushSampleResults(hydrated));
      }
    } catch (e, st) {
      debugPrint('[detection] sample ${job.sampleId} failed: $e\n$st');
      notifier.value = DetectionFailed('$e');
      notifyListeners();
    }
  }

  /// Rebuild the combined segmentation overlay from the supplied [instances]
  /// and persist it for [sampleId]. Called after the user edits, adds, or
  /// deletes an instance so the viewer's "Segment" mode reflects the new
  /// masks instead of the original detection-time render.
  ///
  /// Returns the encoded PNG (or null when [instances] is empty).
  Future<Uint8List?> rebuildSegmentationOverlay({
    required int sampleId,
    required int workingW,
    required int workingH,
    required List<SampleInstance> instances,
  }) async {
    final pipe = WheatHeadPipeline.instance;
    final repo = SampleRepository.instance;
    final maskRefs = <({Uint8List mask, int width, int height})>[];
    for (final inst in instances) {
      if (inst.imageWidth != workingW || inst.imageHeight != workingH) {
        continue;
      }
      final decoded = pipe.decodeMaskPng(inst.maskPng);
      maskRefs.add((
        mask: decoded.mask,
        width: decoded.width,
        height: decoded.height,
      ));
    }
    Uint8List? overlayPng;
    if (maskRefs.isNotEmpty) {
      overlayPng = pipe.renderCombinedSegmentationOverlay(
        width: workingW,
        height: workingH,
        masks: maskRefs,
      );
    }
    await repo.saveSegmentationOverlay(sampleId: sampleId, png: overlayPng);
    return overlayPng;
  }

  /// Re-run FHB analysis for an already-detected sample (e.g. after the
  /// user edits an instance mask). Persists the per-instance FHB stats and
  /// the combined disease overlay; returns the updated instances and PNG.
  ///
  /// Exposed publicly so the sample viewer can call it directly without
  /// re-queuing the whole detection pipeline.
  ///
  /// [sourceOverride] is the original full-res capture, decoded once. When
  /// provided, the per-instance disease previews are rendered from it for
  /// crisp tiles. When null we fall back to decoding the working image,
  /// which keeps callers (e.g. the viewer's post-edit refresh) from having
  /// to read the original JPEG every time.
  Future<
      ({
        List<SampleInstance> instances,
        Uint8List? overlayPng,
        Uint8List? segmentationOverlayPng,
      })> runDiseaseAnalysis({
    required DetectionMode mode,
    required Uint8List workingPng,
    required int workingW,
    required int workingH,
    required List<SampleInstance> instances,
    required int sampleId,
    img.Image? sourceOverride,
    void Function(int done, int total)? onProgress,
  }) async {
    final pipe = WheatHeadPipeline.instance;
    final repo = SampleRepository.instance;
    final workingImage = pipe.decodeWorkingImage(workingPng);
    final previewSource = sourceOverride ?? workingImage;
    final reports = <FhbReport>[];
    final updated = <SampleInstance>[];
    final segMaskRefs = <({Uint8List mask, int width, int height})>[];
    for (int i = 0; i < instances.length; i++) {
      final inst = instances[i];
      if (inst.imageWidth != workingW || inst.imageHeight != workingH) {
        updated.add(inst);
        onProgress?.call(i + 1, instances.length);
        continue;
      }
      final mask = pipe.decodeMaskPng(inst.maskPng).mask;
      var report = switch (mode) {
        DetectionMode.wheatFhb => await pipe.analyzeDisease(
            workingImage: workingImage,
            mask: mask,
            maskWidth: workingW,
            maskHeight: workingH,
            bbox: inst.bbox,
          ),
        DetectionMode.grapeLeaf =>
          await GrapeLeafPipeline.instance.analyzeDisease(
            workingImage: workingImage,
            mask: mask,
            maskWidth: workingW,
            maskHeight: workingH,
            bbox: inst.bbox,
          ),
      };
      // Post-analysis refinement (wheat only): drop tiny "other" specks from
      // the spike mask + classification so the segmentation outline users
      // see in "Segment" mode matches what the disease counts were derived
      // from. Mutates `mask` and the report's classification in place.
      if (mode == DetectionMode.wheatFhb) {
        report = await pipe.refineDisease(
          report: report,
          mask: mask,
          bbox: inst.bbox,
        );
      }
      reports.add(report);
      final diseasePreview = pipe.renderDiseasePreview(
        source: previewSource,
        report: report,
        bbox: inst.bbox,
      );
      // Persist the post-refinement mask so the segmentation overlay + the
      // per-instance preview tile reflect the cleanup. Grape skips refine,
      // so its `mask` is unchanged and the re-encode is effectively a no-op.
      final updatedMaskPng = pipe.encodeMaskPng(mask, workingW, workingH);
      final updatedPreviewPng = pipe.renderInstancePreview(
        source: previewSource,
        mask: mask,
        maskWidth: workingW,
        maskHeight: workingH,
        bbox: inst.bbox,
      );
      segMaskRefs.add((mask: mask, width: workingW, height: workingH));
      final enriched = inst.copyWith(
        maskPng: updatedMaskPng,
        previewPng: updatedPreviewPng,
        fhbGreenCount: report.greenCount,
        fhbNecroticCount: report.necroticCount,
        fhbOtherCount: report.otherCount,
        fhbTotalPixels: report.totalPixels,
        fhbRatio: report.fhbRatio,
        fhbSeverity: report.severity,
        diseasePreviewPng: diseasePreview,
        updatedAt: DateTime.now(),
      );
      if (enriched.id != null) {
        await repo.updateInstance(enriched);
      }
      updated.add(enriched);
      onProgress?.call(i + 1, instances.length);
    }

    Uint8List? overlayPng;
    if (reports.isNotEmpty) {
      overlayPng = pipe.renderCombinedDiseaseOverlay(
        width: workingW,
        height: workingH,
        reports: reports,
      );
    }
    await repo.saveDiseaseOverlay(sampleId: sampleId, png: overlayPng);
    Uint8List? segOverlayPng;
    if (segMaskRefs.isNotEmpty) {
      segOverlayPng = pipe.renderCombinedSegmentationOverlay(
        width: workingW,
        height: workingH,
        masks: segMaskRefs,
      );
      await repo.saveSegmentationOverlay(
          sampleId: sampleId, png: segOverlayPng);
    }
    return (
      instances: updated,
      overlayPng: overlayPng,
      segmentationOverlayPng: segOverlayPng,
    );
  }
}

class _PendingJob {
  const _PendingJob({
    required this.sampleId,
    required this.filePath,
    required this.mode,
  });
  final int sampleId;
  final String filePath;
  final DetectionMode mode;
}
