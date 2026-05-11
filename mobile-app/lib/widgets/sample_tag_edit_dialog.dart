import 'package:flutter/material.dart';
import 'package:gopher_eye/model/sample.dart';

/// Plain bag of QR field values for handoff between the dialog and the
/// caller. Each field is the trimmed user input, or null when blank.
class SampleTagDraft {
  const SampleTagDraft({
    this.qrId,
    this.qrLine,
    this.qrRep,
    this.qrLocation,
    this.qrNote,
  });
  final String? qrId;
  final String? qrLine;
  final String? qrRep;
  final String? qrLocation;
  final String? qrNote;
}

/// Dark-themed editor for a sample's QR-derived "tag" fields. Returns a
/// [SampleTagDraft] on save, or null when cancelled. Shared by the instance
/// inspector and the sample viewer so both edit paths look identical.
class SampleTagEditDialog extends StatefulWidget {
  const SampleTagEditDialog({super.key, required this.initial});
  final Sample initial;

  @override
  State<SampleTagEditDialog> createState() => _SampleTagEditDialogState();
}

class _SampleTagEditDialogState extends State<SampleTagEditDialog> {
  late final TextEditingController _idCtrl =
      TextEditingController(text: widget.initial.qrId ?? '');
  late final TextEditingController _lineCtrl =
      TextEditingController(text: widget.initial.qrLine ?? '');
  late final TextEditingController _repCtrl =
      TextEditingController(text: widget.initial.qrRep ?? '');
  late final TextEditingController _locCtrl =
      TextEditingController(text: widget.initial.qrLocation ?? '');
  late final TextEditingController _noteCtrl =
      TextEditingController(text: widget.initial.qrNote ?? '');

  @override
  void dispose() {
    _idCtrl.dispose();
    _lineCtrl.dispose();
    _repCtrl.dispose();
    _locCtrl.dispose();
    _noteCtrl.dispose();
    super.dispose();
  }

  String? _trimmedOrNull(TextEditingController c) {
    final v = c.text.trim();
    return v.isEmpty ? null : v;
  }

  void _save() {
    Navigator.of(context).pop(SampleTagDraft(
      qrId: _trimmedOrNull(_idCtrl),
      qrLine: _trimmedOrNull(_lineCtrl),
      qrRep: _trimmedOrNull(_repCtrl),
      qrLocation: _trimmedOrNull(_locCtrl),
      qrNote: _trimmedOrNull(_noteCtrl),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: const Color(0xFF1E1E1E),
      title: const Text(
        'Edit sample tag',
        style: TextStyle(color: Colors.white),
      ),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _field(_idCtrl, 'ID'),
            const SizedBox(height: 12),
            _field(_lineCtrl, 'Line'),
            const SizedBox(height: 12),
            _field(_repCtrl, 'Rep', keyboardType: TextInputType.number),
            const SizedBox(height: 12),
            _field(_locCtrl, 'Location'),
            const SizedBox(height: 12),
            _field(_noteCtrl, 'Note', maxLines: 3),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        TextButton(
          onPressed: _save,
          child: const Text(
            'Save',
            style: TextStyle(fontWeight: FontWeight.w600),
          ),
        ),
      ],
    );
  }

  Widget _field(
    TextEditingController controller,
    String label, {
    TextInputType? keyboardType,
    int maxLines = 1,
  }) {
    return TextField(
      controller: controller,
      keyboardType: keyboardType,
      maxLines: maxLines,
      style: const TextStyle(color: Colors.white, fontSize: 14),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(color: Colors.white54),
        enabledBorder: const OutlineInputBorder(
          borderSide: BorderSide(color: Colors.white24),
        ),
        focusedBorder: const OutlineInputBorder(
          borderSide: BorderSide(color: Colors.white70),
        ),
        isDense: true,
      ),
    );
  }
}
