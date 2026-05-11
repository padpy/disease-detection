import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:gal/gal.dart';
import 'package:gopher_eye/model/sample_qr_metadata.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:qr_flutter/qr_flutter.dart';

/// Form that turns user-supplied metadata (id, line, rep, location, note)
/// into a QR code. The generated payload is the same JSON format the
/// camera-side scanner consumes, so codes printed/saved here can be
/// re-scanned to attach the metadata to future captures.
class QrCreatorScreen extends StatefulWidget {
  const QrCreatorScreen({super.key});

  @override
  State<QrCreatorScreen> createState() => _QrCreatorScreenState();
}

class _QrCreatorScreenState extends State<QrCreatorScreen> {
  final _idController = TextEditingController();
  final _lineController = TextEditingController();
  final _repController = TextEditingController();
  final _locationController = TextEditingController();
  final _noteController = TextEditingController();
  final _qrKey = GlobalKey();

  bool _saving = false;

  @override
  void dispose() {
    _idController.dispose();
    _lineController.dispose();
    _repController.dispose();
    _locationController.dispose();
    _noteController.dispose();
    super.dispose();
  }

  String? _trimmedOrNull(TextEditingController c) {
    final v = c.text.trim();
    return v.isEmpty ? null : v;
  }

  SampleQrMetadata _buildMetadata() => SampleQrMetadata(
        qrId: _trimmedOrNull(_idController),
        line: _trimmedOrNull(_lineController),
        rep: _trimmedOrNull(_repController),
        location: _trimmedOrNull(_locationController),
        note: _trimmedOrNull(_noteController),
      );

  Future<Uint8List?> _renderQrPng() async {
    final boundary = _qrKey.currentContext?.findRenderObject()
        as RenderRepaintBoundary?;
    if (boundary == null) return null;
    final image = await boundary.toImage(pixelRatio: 3.0);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    return byteData?.buffer.asUint8List();
  }

  Future<void> _saveToGallery() async {
    if (_saving) return;
    setState(() => _saving = true);
    try {
      final png = await _renderQrPng();
      if (png == null) {
        _showSnack('Could not render QR code.');
        return;
      }
      final tmp = await getTemporaryDirectory();
      final filename =
          'gopher_eye_qr_${DateTime.now().millisecondsSinceEpoch}.png';
      final path = p.join(tmp.path, filename);
      await File(path).writeAsBytes(png);
      await Gal.putImage(path);
      _showSnack('QR code saved to Photos');
    } catch (e, st) {
      debugPrint('[qr] save failed: $e\n$st');
      _showSnack('Save failed: $e');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  void _showSnack(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final metadata = _buildMetadata();
    final payload = metadata.isEmpty ? null : metadata.encode();
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        title: const Text(
          'Create sample QR',
          style: TextStyle(color: Colors.white),
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
          children: [
            const Text(
              'Fill in any combination of fields. Codes are scannable by the '
              'camera screen and apply the metadata to your next captures.',
              style: TextStyle(color: Colors.white70, fontSize: 13),
            ),
            const SizedBox(height: 16),
            _Field(
              label: 'ID',
              controller: _idController,
              hint: 'e.g. plot-23',
              onChanged: (_) => setState(() {}),
            ),
            _Field(
              label: 'Line',
              controller: _lineController,
              hint: 'e.g. variety / breeding line',
              onChanged: (_) => setState(() {}),
            ),
            _Field(
              label: 'Rep',
              controller: _repController,
              hint: 'e.g. 1',
              onChanged: (_) => setState(() {}),
            ),
            _Field(
              label: 'Location',
              controller: _locationController,
              hint: 'e.g. North field, block 4',
              onChanged: (_) => setState(() {}),
            ),
            _Field(
              label: 'Note',
              controller: _noteController,
              hint: 'Free text — kept with the sample',
              onChanged: (_) => setState(() {}),
              maxLines: 3,
            ),
            const SizedBox(height: 16),
            const _SectionLabel('Preview'),
            Center(
              child: Container(
                margin: const EdgeInsets.symmetric(vertical: 8),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: payload == null
                    ? const SizedBox(
                        width: 220,
                        height: 220,
                        child: Center(
                          child: Text(
                            'Add at least one field\nto generate a QR code.',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: Colors.black54),
                          ),
                        ),
                      )
                    : RepaintBoundary(
                        key: _qrKey,
                        child: Container(
                          color: Colors.white,
                          padding: const EdgeInsets.all(8),
                          child: QrImageView(
                            data: payload,
                            version: QrVersions.auto,
                            size: 240,
                            backgroundColor: Colors.white,
                            errorCorrectionLevel: QrErrorCorrectLevel.M,
                          ),
                        ),
                      ),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: payload == null || _saving ? null : _saveToGallery,
              icon: _saving
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.save_alt),
              label: const Text('Save QR to Photos'),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

class _Field extends StatelessWidget {
  const _Field({
    required this.label,
    required this.controller,
    this.hint,
    this.maxLines = 1,
    this.onChanged,
  });

  final String label;
  final TextEditingController controller;
  final String? hint;
  final int maxLines;
  final ValueChanged<String>? onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: TextField(
        controller: controller,
        maxLines: maxLines,
        autocorrect: false,
        onChanged: onChanged,
        style: const TextStyle(color: Colors.white),
        decoration: InputDecoration(
          labelText: label,
          labelStyle: const TextStyle(color: Colors.white60),
          hintText: hint,
          hintStyle: const TextStyle(color: Colors.white30),
          enabledBorder: const OutlineInputBorder(
            borderSide: BorderSide(color: Colors.white24),
          ),
          focusedBorder: const OutlineInputBorder(
            borderSide: BorderSide(color: Colors.white),
          ),
        ),
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 12, 4, 4),
      child: Text(
        text.toUpperCase(),
        style: const TextStyle(
          color: Colors.white60,
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.8,
        ),
      ),
    );
  }
}
