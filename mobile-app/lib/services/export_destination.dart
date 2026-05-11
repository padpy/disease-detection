import 'package:flutter/material.dart';
import 'package:gopher_eye/services/export_service.dart';
import 'package:share_plus/share_plus.dart';

/// Where the user wants the export bundle delivered. Both routes ultimately
/// hand the bytes to the platform — the difference is what the share sheet
/// is hinted to surface.
enum ExportDestination {
  /// Save to the device's filesystem. On iOS the user picks "Save to Files";
  /// on Android the chooser includes Files / Downloads.
  files,

  /// Hand the artifact to Google Drive via the share sheet. Drive must be
  /// installed; otherwise the user can still pick another target.
  googleDrive,
}

extension ExportDestinationLabel on ExportDestination {
  String get label => switch (this) {
        ExportDestination.files => 'Save to phone files',
        ExportDestination.googleDrive => 'Upload to Google Drive',
      };

  IconData get icon => switch (this) {
        ExportDestination.files => Icons.folder_outlined,
        ExportDestination.googleDrive => Icons.cloud_upload_outlined,
      };
}

/// Hand [artifact] off to the platform sharing UI. Both destinations route
/// through the system share sheet so users can pick Files, Drive, AirDrop,
/// or any other registered handler — the destination they picked in the
/// in-app dialog is just a hint for the prompt text.
Future<bool> deliverExport({
  required ExportArtifact artifact,
  required ExportDestination destination,
  required Rect? sharePositionOrigin,
}) async {
  final result = await Share.shareXFiles(
    [XFile(artifact.file.path, mimeType: artifact.mimeType)],
    fileNameOverrides: [artifact.suggestedName],
    subject: 'Gopher Eye export',
    text: switch (destination) {
      ExportDestination.files =>
        'Save the attached Gopher Eye export to Files.',
      ExportDestination.googleDrive =>
        'Upload the attached Gopher Eye export to Google Drive.',
    },
    sharePositionOrigin: sharePositionOrigin,
  );
  return result.status == ShareResultStatus.success;
}

/// Best-effort cleanup of the on-disk artifact once the share sheet has had
/// a chance to consume it. iOS holds the file open during the sheet's
/// lifetime, so we delete only after [deliverExport] returns.
Future<void> cleanupExport(ExportArtifact artifact) async {
  try {
    if (await artifact.file.exists()) {
      await artifact.file.delete();
    }
  } catch (_) {
    // Best-effort — temp dir is cleared by the OS eventually.
  }
}
