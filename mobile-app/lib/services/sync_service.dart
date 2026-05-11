import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:gopher_eye/model/detection_mode.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/model/sample_instance.dart';
import 'package:gopher_eye/services/api_client.dart';
import 'package:gopher_eye/services/app_settings.dart';
import 'package:gopher_eye/services/sample_repository.dart';

/// Pushes local samples (and their instances + blobs) to the configured
/// server, and pulls remote samples down so the user can see backups from
/// other devices. Each operation is best-effort — failures are logged but
/// don't surface to the UI unless the caller awaits them, since most pushes
/// happen as fire-and-forget side effects of capture/detection.
class SyncService {
  SyncService._();
  static final SyncService instance = SyncService._();

  /// True iff the user has both enabled sync AND configured a server URL.
  Future<bool> isEnabled() async {
    if (!await AppSettings.getSyncEnabled()) return false;
    final url = await AppSettings.getServerUrl();
    return url != null && url.isNotEmpty;
  }

  /// Push the entire sample (image + metadata) to the server. Records the
  /// returned remote id so subsequent pushes (e.g. when detection completes)
  /// target the same remote row. No-op if sync is disabled or the sample is
  /// already linked to a remote id.
  Future<int?> pushSample(Sample sample) async {
    if (sample.id == null) return null;
    if (!await isEnabled()) return null;
    final repo = SampleRepository.instance;
    final existingRemote = await repo.sampleRemoteId(sample.id!);
    if (existingRemote != null) return existingRemote;

    final file = File(sample.filePath);
    final user = await AppSettings.getUserName();
    try {
      // Existence check + read are racy with cleanup tasks (e.g. user wipes
      // the gallery); fold both into one try/catch so a missing-file
      // FileSystemException doesn't bubble out of the fire-and-forget caller.
      if (!await file.exists()) {
        debugPrint('[sync] skipping push for ${sample.id} — file missing');
        return null;
      }
      final bytes = await file.readAsBytes();
      final remote = await ApiClient.instance.createSample(
        imageBytes: bytes,
        filename: file.uri.pathSegments.last,
        metadata: {
          'taken_at': sample.takenAt.millisecondsSinceEpoch,
          'latitude': sample.latitude,
          'longitude': sample.longitude,
          'accuracy': sample.accuracy,
          'detection_mode': sample.detectionMode.id,
          if (user.isNotEmpty) 'user': user,
          if (sample.collectionId != null) 'collection_id': sample.collectionId,
          if (sample.qrId != null) 'qr_id': sample.qrId,
          if (sample.qrLine != null) 'qr_line': sample.qrLine,
          if (sample.qrRep != null) 'qr_rep': sample.qrRep,
          if (sample.qrLocation != null) 'qr_location': sample.qrLocation,
          if (sample.qrNote != null) 'qr_note': sample.qrNote,
        },
      );
      final remoteId = remote['id'] as int?;
      if (remoteId != null) {
        await repo.setSampleRemoteId(sample.id!, remoteId);
      }
      return remoteId;
    } catch (e, st) {
      debugPrint('[sync] pushSample(${sample.id}) failed: $e\n$st');
      return null;
    }
  }

  /// Push detection results (instances, working image, overlays) for [sample]
  /// to the server. Pushes the sample row first if not already linked.
  Future<void> pushSampleResults(Sample sample) async {
    if (sample.id == null) return;
    if (!await isEnabled()) return;
    final repo = SampleRepository.instance;
    var remoteId = await repo.sampleRemoteId(sample.id!);
    remoteId ??= await pushSample(sample);
    if (remoteId == null) return;

    try {
      // Working image blob (with width/height metadata) so the server can
      // re-render overlays at the same resolution mobile is using.
      final wi = await repo.loadWorkingImage(sample.id!);
      if (wi != null) {
        await ApiClient.instance.putSampleBlob(
          remoteId,
          'working_image_png',
          wi.png,
          width: wi.width,
          height: wi.height,
        );
      }
      final disease = await repo.loadDiseaseOverlay(sample.id!);
      if (disease != null) {
        await ApiClient.instance
            .putSampleBlob(remoteId, 'disease_overlay_png', disease);
      }
      final segmentation = await repo.loadSegmentationOverlay(sample.id!);
      if (segmentation != null) {
        await ApiClient.instance
            .putSampleBlob(remoteId, 'segmentation_overlay_png', segmentation);
      }

      final instances = await repo.listInstances(sample.id!);
      if (instances.isNotEmpty) {
        final payloads = instances
            .map((inst) => _instanceToPayload(inst))
            .toList(growable: false);
        await ApiClient.instance.replaceInstances(remoteId, payloads);
      }
    } catch (e, st) {
      debugPrint('[sync] pushSampleResults(${sample.id}) failed: $e\n$st');
    }
  }

  /// Fetch the list of samples on the server and insert any rows we don't
  /// already have locally (matched by [Sample.remote_id]). Returns the count
  /// of newly-inserted local rows.
  Future<int> pullSamples({String? user}) async {
    if (await AppSettings.getServerUrl() == null) {
      throw const ApiNotConfiguredException();
    }
    final repo = SampleRepository.instance;
    final list = await ApiClient.instance.listSamples(user: user);
    int inserted = 0;
    for (final remote in list) {
      final remoteId = remote['id'] as int?;
      if (remoteId == null) continue;
      try {
        final existing = await repo.findSampleByRemoteId(remoteId);
        if (existing != null) continue;

        final imageBytes = await _safeFetchSource(remoteId);
        if (imageBytes == null) continue;
        final filePath = await _writeIncomingImage(remoteId, imageBytes);
        final takenAt = DateTime.fromMillisecondsSinceEpoch(
            (remote['taken_at'] as int?) ??
                DateTime.now().millisecondsSinceEpoch);
        final mode = DetectionMode.fromId(remote['detection_mode'] as String?);
        await repo.insertSampleFromRemote(
          remoteId: remoteId,
          filePath: filePath,
          takenAt: takenAt,
          detectionMode: mode,
          latitude: (remote['latitude'] as num?)?.toDouble(),
          longitude: (remote['longitude'] as num?)?.toDouble(),
          accuracy: (remote['accuracy'] as num?)?.toDouble(),
        );
        inserted += 1;
      } catch (e) {
        debugPrint('[sync] pull failed for remote $remoteId: $e');
      }
    }
    return inserted;
  }

  Future<Uint8List?> _safeFetchSource(int remoteId) async {
    try {
      return await ApiClient.instance.getSampleSource(remoteId);
    } catch (e) {
      debugPrint('[sync] pull source failed for $remoteId: $e');
      return null;
    }
  }

  Future<String> _writeIncomingImage(int remoteId, Uint8List bytes) async {
    if (bytes.isEmpty) {
      throw StateError(
          'Server returned empty body for sample source $remoteId');
    }
    final docs = Directory.systemTemp;
    final dir = Directory('${docs.path}/gopher_eye_pulled');
    if (!await dir.exists()) await dir.create(recursive: true);
    final path = '${dir.path}/remote_$remoteId.jpg';
    await File(path).writeAsBytes(bytes, flush: true);
    return path;
  }

  Map<String, dynamic> _instanceToPayload(SampleInstance inst) {
    return {
      'idx': inst.idx,
      'bbox': {
        'left': inst.bbox.left,
        'top': inst.bbox.top,
        'right': inst.bbox.right,
        'bottom': inst.bbox.bottom,
      },
      'centroid': {'x': inst.centroid.dx, 'y': inst.centroid.dy},
      'score': inst.score,
      'image_w': inst.imageWidth,
      'image_h': inst.imageHeight,
      'mask_png': base64Encode(inst.maskPng),
      'preview_png': base64Encode(inst.previewPng),
      if (inst.fhbGreenCount != null) 'fhb_green': inst.fhbGreenCount,
      if (inst.fhbNecroticCount != null) 'fhb_necrotic': inst.fhbNecroticCount,
      if (inst.fhbOtherCount != null) 'fhb_other': inst.fhbOtherCount,
      if (inst.fhbTotalPixels != null) 'fhb_total': inst.fhbTotalPixels,
      if (inst.fhbRatio != null) 'fhb_ratio': inst.fhbRatio,
      if (inst.fhbSeverity != null) 'fhb_severity': inst.fhbSeverity,
      if (inst.diseasePreviewPng != null)
        'disease_preview_png': base64Encode(inst.diseasePreviewPng!),
      'created_at': inst.createdAt.millisecondsSinceEpoch,
      'updated_at': inst.updatedAt.millisecondsSinceEpoch,
    };
  }
}
