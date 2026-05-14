import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gal/gal.dart';
import 'package:gopher_eye/model/collection.dart';
import 'package:gopher_eye/model/detection_mode.dart';
import 'package:gopher_eye/model/sample.dart';
import 'package:gopher_eye/model/sample_qr_metadata.dart';
import 'package:gopher_eye/screens/chatbot_screen.dart';
import 'package:gopher_eye/screens/collection_picker_screen.dart';
import 'package:gopher_eye/screens/qr_scanner_screen.dart';
import 'package:gopher_eye/screens/samples_screen.dart';
import 'package:gopher_eye/screens/settings_screen.dart';
import 'package:gopher_eye/services/detection_service.dart';
import 'package:gopher_eye/services/location_service.dart';
import 'package:gopher_eye/services/sample_repository.dart';
import 'package:gopher_eye/services/sync_service.dart';
import 'package:gopher_eye/widgets/siri_shimmer.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:photo_manager/photo_manager.dart';

/// Cross-instance memory of the user's last selected detection mode so the
/// chip stays where they left it as they swipe between camera + samples.
/// In-memory only — resets on app restart.
DetectionMode _lastDetectionMode = DetectionMode.wheatFhb;

/// In-memory cache of the last scanned QR metadata so swiping between camera
/// and samples doesn't drop the user's "active sample tag." Cleared on app
/// restart and via the chip's clear button.
SampleQrMetadata? _activeQrMetadata;

/// Active capture session. While set, every photo taken from the camera
/// screen is tagged with this collection's id. Lives at module scope (rather
/// than per-state) so swiping between samples + camera doesn't reset the
/// session — only an explicit "None" pick or app restart does.
Collection? _activeCollection;

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, this.onOpenSamples});

  final VoidCallback? onOpenSamples;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  int _cameraIndex = 0;
  bool _permissionGranted = false;
  bool _initialized = false;
  bool _switching = false;
  FlashMode _flashMode = FlashMode.off;
  bool _flashOverlay = false;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  double _zoom = 1.0;
  double _zoomStart = 1.0;
  bool _zoomIndicatorVisible = false;
  Timer? _zoomIndicatorTimer;
  Uint8List? _galleryThumb;
  bool _picking = false;
  DetectionMode _mode = _lastDetectionMode;
  SampleQrMetadata? _qrMetadata = _activeQrMetadata;
  Collection? _collection = _activeCollection;
  bool _chatbotMode = false;
  bool _speedDialOpen = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _setup();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _zoomIndicatorTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      controller.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initController(_cameras[_cameraIndex]);
    }
  }

  Future<void> _setup() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      if (mounted) setState(() => _permissionGranted = false);
      return;
    }
    _cameras = await availableCameras();
    if (_cameras.isEmpty) return;
    if (mounted) setState(() => _permissionGranted = true);
    await _initController(_cameras[_cameraIndex]);
    unawaited(_loadGalleryThumb());
  }

  Future<void> _loadGalleryThumb() async {
    try {
      final ps = await PhotoManager.requestPermissionExtend();
      if (!ps.hasAccess) return;
      final paths = await PhotoManager.getAssetPathList(
        type: RequestType.image,
        onlyAll: true,
      );
      if (paths.isEmpty) return;
      final assets = await paths.first.getAssetListPaged(page: 0, size: 1);
      if (assets.isEmpty) return;
      final thumb = await assets.first.thumbnailDataWithSize(
        const ThumbnailSize(160, 160),
      );
      if (mounted) setState(() => _galleryThumb = thumb);
    } catch (e, st) {
      debugPrint('[gallery] thumb load failed: $e\n$st');
    }
  }

  Future<void> _initController(CameraDescription description) async {
    final previous = _controller;
    if (mounted) {
      setState(() {
        _initialized = false;
        _controller = null;
      });
    }
    await previous?.dispose();

    // Capture at the sensor's native resolution. Downstream pipelines
    // (wheat YOLO26, SAM, server-side detection) re-scale to their own
    // working frames; downsampling at capture time was throwing away
    // detail that the wheat-head detector actually relies on, and the
    // server explicitly wants the original-resolution upload.
    final next = CameraController(
      description,
      ResolutionPreset.max,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    next.addListener(() {
      if (mounted && _controller == next) setState(() {});
    });
    double minZoom = 1.0;
    double maxZoom = 1.0;
    double initialZoom = 1.0;
    try {
      await next.initialize();
      await next.lockCaptureOrientation(DeviceOrientation.portraitUp);
      await next.setFlashMode(_flashMode);
      // Use what the active camera actually reports — forcing a 0.5x floor
      // would re-enable the preset chip on hardware that physically can't
      // zoom that wide (e.g. iOS opens only the wide-angle camera, min 1.0),
      // and setZoomLevel(0.5) would silently throw.
      minZoom = await next.getMinZoomLevel();
      maxZoom = await next.getMaxZoomLevel();
      initialZoom = minZoom.clamp(minZoom, maxZoom);
      await next.setZoomLevel(initialZoom);
    } catch (_) {
      await next.dispose();
      return;
    }
    if (!mounted) {
      await next.dispose();
      return;
    }
    setState(() {
      _controller = next;
      _initialized = next.value.isInitialized;
      _minZoom = minZoom;
      _maxZoom = maxZoom;
      _zoom = initialZoom;
    });
  }

  void _showZoomIndicator() {
    _zoomIndicatorTimer?.cancel();
    if (!_zoomIndicatorVisible) {
      setState(() => _zoomIndicatorVisible = true);
    }
    _zoomIndicatorTimer = Timer(const Duration(seconds: 3), () {
      if (mounted) setState(() => _zoomIndicatorVisible = false);
    });
  }

  void _handleScaleStart(ScaleStartDetails details) {
    _zoomStart = _zoom;
    _showZoomIndicator();
  }

  Future<void> _handleScaleUpdate(ScaleUpdateDetails details) async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (details.pointerCount < 2) return;
    final next = (_zoomStart * details.scale).clamp(_minZoom, _maxZoom);
    if ((next - _zoom).abs() < 0.01) return;
    try {
      await controller.setZoomLevel(next);
      if (mounted) {
        setState(() => _zoom = next);
        _showZoomIndicator();
      }
    } catch (_) {}
  }

  Future<void> _setZoom(double level) async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    final clamped = level.clamp(_minZoom, _maxZoom);
    try {
      await controller.setZoomLevel(clamped);
      if (mounted) {
        setState(() => _zoom = clamped);
        _showZoomIndicator();
      }
    } catch (_) {}
  }

  Future<void> _flipCamera() async {
    if (_switching || _cameras.length < 2) return;
    _switching = true;
    try {
      _cameraIndex = (_cameraIndex + 1) % _cameras.length;
      await _initController(_cameras[_cameraIndex]);
    } finally {
      _switching = false;
    }
  }

  Future<void> _toggleFlash() async {
    final controller = _controller;
    if (controller == null) return;
    final next = _flashMode == FlashMode.off ? FlashMode.torch : FlashMode.off;
    try {
      await controller.setFlashMode(next);
      if (mounted) setState(() => _flashMode = next);
    } catch (_) {}
  }

  Future<void> _capture() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (controller.value.isTakingPicture) return;
    try {
      final file = await controller.takePicture();
      if (!mounted) return;
      setState(() => _flashOverlay = true);
      Future.delayed(const Duration(milliseconds: 120), () {
        if (mounted) setState(() => _flashOverlay = false);
      });

      if (_chatbotMode) {
        await _routeToChatbot(file);
        return;
      }

      final position = await LocationService.tryGetCurrent();
      debugPrint('[capture] position=$position');
      final qr = _qrMetadata;
      final sample = await SampleRepository.instance.saveCapture(
        image: file,
        position: position,
        detectionMode: _mode,
        collectionId: _collection?.id,
        qrId: qr?.qrId,
        qrLine: qr?.line,
        qrRep: qr?.rep,
        qrLocation: qr?.location,
        qrNote: qr?.note,
      );
      debugPrint('[capture] saved sample id=${sample.id} path=${sample.filePath}');
      try {
        await Gal.putImage(sample.filePath);
      } catch (e, st) {
        debugPrint('[capture] Gal.putImage failed: $e\n$st');
      }
      if (sample.id != null) {
        DetectionService.instance.enqueue(
          sampleId: sample.id!,
          filePath: sample.filePath,
          mode: _mode,
        );
        unawaited(SyncService.instance.pushSample(sample));
      }
      _showSnack('Saved sample #${sample.id} · ${_mode.label}');
      unawaited(_loadGalleryThumb());
    } on CameraException catch (e, st) {
      debugPrint('[capture] CameraException: $e\n$st');
      _showSnack('Camera error: ${e.code}');
    } catch (e, st) {
      debugPrint('[capture] save failed: $e\n$st');
      _showSnack('Save failed: $e');
    }
  }

  void _showSnack(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Text(message),
          duration: const Duration(seconds: 2),
          behavior: SnackBarBehavior.floating,
        ),
      );
  }

  void _toggleChatbotMode() {
    setState(() => _chatbotMode = !_chatbotMode);
    _showSnack(
      _chatbotMode
          ? 'Chatbot mode on — capture or pick a photo to chat about it.'
          : 'Chatbot mode off',
    );
  }

  void _openSettings() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const SettingsScreen()),
    );
  }

  void _openSamples() {
    final swipe = widget.onOpenSamples;
    if (swipe != null) {
      swipe();
      return;
    }
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const SamplesScreen()),
    );
  }

  Future<void> _pickCollection() async {
    final result =
        await Navigator.of(context).push<CollectionPickResult>(
      MaterialPageRoute(
        builder: (_) => CollectionPickerScreen(
          activeCollectionId: _collection?.id,
        ),
      ),
    );
    if (result == null || !mounted) return;
    setState(() {
      _collection = result.collection;
      _activeCollection = result.collection;
    });
    if (result.cleared) {
      _showSnack('Collection cleared');
    } else if (result.collection != null) {
      _showSnack('Collection: ${result.collection!.name}');
    }
  }

  void _clearCollection() {
    setState(() {
      _collection = null;
      _activeCollection = null;
    });
    _showSnack('Collection cleared');
  }

  Future<void> _scanQr() async {
    final result = await Navigator.of(context).push<SampleQrMetadata>(
      MaterialPageRoute(builder: (_) => const QrScannerScreen()),
    );
    if (result == null || !mounted) return;
    setState(() {
      _qrMetadata = result;
      _activeQrMetadata = result;
    });
    _showSnack('Tag attached: ${result.displayLabel ?? 'QR metadata'}');
  }

  void _clearQr() {
    setState(() {
      _qrMetadata = null;
      _activeQrMetadata = null;
    });
    _showSnack('Cleared sample tag');
  }

  Future<void> _pickMode() async {
    final picked = await showModalBottomSheet<DetectionMode>(
      context: context,
      backgroundColor: Colors.grey[900],
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Detection mode',
                  style: TextStyle(color: Colors.white, fontSize: 14),
                ),
              ),
            ),
            for (final m in DetectionMode.values)
              ListTile(
                leading: Icon(
                  m == _mode
                      ? Icons.radio_button_checked
                      : Icons.radio_button_unchecked,
                  color: Colors.white,
                ),
                title: Text(
                  m.longLabel,
                  style: const TextStyle(color: Colors.white),
                ),
                subtitle: Text(
                  m.hasAutoDetection
                      ? 'Auto-detects ${m.instanceNounPlural} on capture'
                      : 'Tap + in the sample viewer to add ${m.instanceNounPlural} manually',
                  style: const TextStyle(color: Colors.white54, fontSize: 12),
                ),
                onTap: () => Navigator.of(ctx).pop(m),
              ),
          ],
        ),
      ),
    );
    if (picked != null && mounted) {
      setState(() {
        _mode = picked;
        _lastDetectionMode = picked;
      });
    }
  }

  Future<void> _pickFromGallery() async {
    if (_picking) return;
    _picking = true;
    try {
      final picker = ImagePicker();

      // Chatbot expects a single image to converse about; bulk would have no
      // sensible target screen. Stay on the single-pick path there.
      if (_chatbotMode) {
        final picked = await picker.pickImage(source: ImageSource.gallery);
        if (picked == null) return;
        await _routeToChatbot(picked);
        return;
      }

      final picked = await picker.pickMultiImage();
      if (picked.isEmpty) return;

      final qr = _qrMetadata;
      final mode = _mode;
      final collectionId = _collection?.id;
      var success = 0;
      var failed = 0;
      Sample? lastSample;
      for (final file in picked) {
        try {
          final sample = await SampleRepository.instance.saveCapture(
            image: file,
            position: null,
            detectionMode: mode,
            collectionId: collectionId,
            qrId: qr?.qrId,
            qrLine: qr?.line,
            qrRep: qr?.rep,
            qrLocation: qr?.location,
            qrNote: qr?.note,
          );
          if (sample.id != null) {
            DetectionService.instance.enqueue(
              sampleId: sample.id!,
              filePath: sample.filePath,
              mode: mode,
            );
            unawaited(SyncService.instance.pushSample(sample));
          }
          lastSample = sample;
          success++;
        } catch (e, st) {
          debugPrint('[pick] one of bulk failed: $e\n$st');
          failed++;
        }
      }

      if (success == 1 && failed == 0 && lastSample != null) {
        _showSnack('Imported sample #${lastSample.id} · ${mode.label}');
      } else if (failed == 0) {
        _showSnack('Imported $success samples · ${mode.label}');
      } else if (success == 0) {
        _showSnack('Import failed for all $failed photos');
      } else {
        _showSnack('Imported $success · $failed failed');
      }
      unawaited(_loadGalleryThumb());
    } catch (e, st) {
      debugPrint('[pick] failed: $e\n$st');
      _showSnack('Import failed: $e');
    } finally {
      _picking = false;
    }
  }

  /// Read [file], normalize to PNG bytes (resized to 1024 longest-edge so
  /// chat payloads stay reasonable), and push the chatbot screen with the
  /// result. Used by both capture + gallery flows when chatbot mode is on.
  Future<void> _routeToChatbot(XFile file) async {
    try {
      final bytes = await _prepareChatImage(file);
      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ChatbotScreen(imageBytes: bytes),
        ),
      );
    } catch (e, st) {
      debugPrint('[chatbot] route failed: $e\n$st');
      _showSnack('Could not open chat: $e');
    }
  }

  Future<Uint8List> _prepareChatImage(XFile file) async {
    final raw = await file.readAsBytes();
    final decoded = img.decodeImage(raw);
    if (decoded == null) return raw;
    const longestEdge = 1024;
    final maxSide = math.max(decoded.width, decoded.height);
    final resized = maxSide > longestEdge
        ? img.copyResize(
            decoded,
            width: decoded.width >= decoded.height ? longestEdge : null,
            height: decoded.height > decoded.width ? longestEdge : null,
          )
        : decoded;
    return Uint8List.fromList(img.encodePng(resized));
  }

  @override
  Widget build(BuildContext context) {
    if (!_permissionGranted) {
      return _PermissionPrompt(onRetry: _setup);
    }
    final controller = _controller;
    if (!_initialized || controller == null) {
      return const ColoredBox(
        color: Colors.black,
        child: Center(
          child: CircularProgressIndicator(color: Colors.white),
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        Positioned.fill(
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onScaleStart: _handleScaleStart,
            onScaleUpdate: _handleScaleUpdate,
            child: SiriShimmer(
              active: _chatbotMode,
              child: _CameraPreview(controller: controller),
            ),
          ),
        ),
        if (_flashOverlay)
          const Positioned.fill(child: ColoredBox(color: Colors.white)),
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Column(
              children: [
                _TopBar(
                  onRecords: _openSamples,
                  mode: _mode,
                  onPickMode: _pickMode,
                  collectionActive: _collection != null,
                  onPickCollection: _pickCollection,
                  chatbotActive: _chatbotMode,
                  onToggleChatbot: _toggleChatbotMode,
                  moreActive: _speedDialOpen,
                  onMoreTap: () =>
                      setState(() => _speedDialOpen = !_speedDialOpen),
                ),
                if (_collection != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: _CollectionChip(
                      collection: _collection!,
                      onClear: _clearCollection,
                    ),
                  ),
                if (_qrMetadata != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: _QrTagChip(
                      metadata: _qrMetadata!,
                      onClear: _clearQr,
                    ),
                  ),
                if (_chatbotMode)
                  const Padding(
                    padding: EdgeInsets.only(top: 8),
                    child: _ChatbotModeChip(),
                  ),
                const Spacer(),
                _LensPresetBar(
                  zoom: _zoom,
                  minZoom: _minZoom,
                  maxZoom: _maxZoom,
                  onSelect: _setZoom,
                ),
                _BottomBar(
                  onCapture: _capture,
                  onPickGallery: _pickFromGallery,
                  galleryThumb: _galleryThumb,
                  zoom: _zoom,
                  zoomIndicatorVisible: _zoomIndicatorVisible,
                  onScanQr: _scanQr,
                ),
              ],
            ),
          ),
        ),
        if (_speedDialOpen)
          _SpeedDial(
            flashOn: _flashMode != FlashMode.off,
            onDismiss: () => setState(() => _speedDialOpen = false),
            onFlash: () {
              _toggleFlash();
            },
            onFlip: () {
              setState(() => _speedDialOpen = false);
              _flipCamera();
            },
            onSettings: () {
              setState(() => _speedDialOpen = false);
              _openSettings();
            },
          ),
      ],
    );
  }
}

class _CameraPreview extends StatelessWidget {
  const _CameraPreview({required this.controller});
  final CameraController controller;

  @override
  Widget build(BuildContext context) {
    final previewSize = controller.value.previewSize;
    if (previewSize == null) {
      return const ColoredBox(color: Colors.black);
    }
    // previewSize is reported in sensor (landscape) coordinates:
    // .width is the long side, .height is the short side. Swap when the
    // device is in portrait so FittedBox sees the on-screen aspect ratio.
    final isPortrait =
        MediaQuery.orientationOf(context) == Orientation.portrait;
    final displayWidth = isPortrait ? previewSize.height : previewSize.width;
    final displayHeight = isPortrait ? previewSize.width : previewSize.height;
    return ClipRect(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: displayWidth,
          height: displayHeight,
          child: CameraPreview(controller),
        ),
      ),
    );
  }
}

class _TopBar extends StatelessWidget {
  const _TopBar({
    required this.onRecords,
    required this.mode,
    required this.onPickMode,
    required this.collectionActive,
    required this.onPickCollection,
    required this.chatbotActive,
    required this.onToggleChatbot,
    required this.moreActive,
    required this.onMoreTap,
  });

  final VoidCallback onRecords;
  final DetectionMode mode;
  final VoidCallback onPickMode;

  /// When true, the collection button is rendered in its "engaged" style.
  /// Tap opens the picker either way.
  final bool collectionActive;
  final VoidCallback onPickCollection;

  /// When true, the chat-bubble button is rendered in its "engaged" style and
  /// the camera screen overlays a Siri-style shimmer around the preview.
  final bool chatbotActive;
  final VoidCallback onToggleChatbot;

  /// True while the speed-dial overlay (flash / flip / settings) is open. The
  /// trigger button uses the highlighted style so users can see at a glance
  /// that tapping again will close it.
  final bool moreActive;
  final VoidCallback onMoreTap;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        _CircleIconButton(
          icon: Icons.photo_library_outlined,
          onTap: onRecords,
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Center(
            child: _ModeChip(mode: mode, onTap: onPickMode),
          ),
        ),
        const SizedBox(width: 8),
        _CircleIconButton(
          icon: Icons.collections_bookmark_outlined,
          onTap: onPickCollection,
          highlighted: collectionActive,
        ),
        const SizedBox(width: 8),
        _CircleIconButton(
          icon: Icons.smart_toy_outlined,
          onTap: onToggleChatbot,
          highlighted: chatbotActive,
        ),
        const SizedBox(width: 8),
        _CircleIconButton(
          icon: moreActive ? Icons.close : Icons.more_horiz,
          onTap: onMoreTap,
          highlighted: moreActive,
        ),
      ],
    );
  }
}

/// Floating column anchored under the top-bar `[+]` trigger that hosts the
/// less-used controls (flash, camera flip, settings). Rendered as a Stack
/// sibling so it overlays the camera preview; a transparent full-screen scrim
/// catches taps outside the column to dismiss.
class _SpeedDial extends StatefulWidget {
  const _SpeedDial({
    required this.flashOn,
    required this.onDismiss,
    required this.onFlash,
    required this.onFlip,
    required this.onSettings,
  });

  final bool flashOn;
  final VoidCallback onDismiss;
  final VoidCallback onFlash;
  final VoidCallback onFlip;
  final VoidCallback onSettings;

  @override
  State<_SpeedDial> createState() => _SpeedDialState();
}

class _SpeedDialState extends State<_SpeedDial>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 220),
    )..forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final safeTop = MediaQuery.of(context).padding.top;
    // Top of safe area + the column's vertical padding (8) + top-bar height
    // (40) + small gap (8). Keep this in sync with the SafeArea/Padding in
    // _CameraScreenState.build.
    final columnTop = safeTop + 8 + 40 + 8;

    final children = <Widget>[
      _SpeedDialChild(
        index: 0,
        controller: _controller,
        icon: widget.flashOn ? Icons.flash_on : Icons.flash_off,
        onTap: widget.onFlash,
      ),
      _SpeedDialChild(
        index: 1,
        controller: _controller,
        icon: Icons.flip_camera_ios,
        onTap: widget.onFlip,
      ),
      _SpeedDialChild(
        index: 2,
        controller: _controller,
        icon: Icons.settings,
        onTap: widget.onSettings,
      ),
    ];

    return Stack(
      children: [
        Positioned.fill(
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: widget.onDismiss,
            child: const SizedBox.shrink(),
          ),
        ),
        Positioned(
          top: columnTop,
          right: 16,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              for (var i = 0; i < children.length; i++) ...[
                if (i > 0) const SizedBox(height: 8),
                children[i],
              ],
            ],
          ),
        ),
      ],
    );
  }
}

class _SpeedDialChild extends StatelessWidget {
  const _SpeedDialChild({
    required this.index,
    required this.controller,
    required this.icon,
    required this.onTap,
  });

  final int index;
  final AnimationController controller;
  final IconData icon;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final start = (index * 0.12).clamp(0.0, 1.0);
    final end = (start + 0.7).clamp(0.0, 1.0);
    final animation = CurvedAnimation(
      parent: controller,
      curve: Interval(start, end, curve: Curves.easeOut),
    );
    return AnimatedBuilder(
      animation: animation,
      builder: (context, child) {
        return Opacity(
          opacity: animation.value,
          child: Transform.translate(
            offset: Offset(0, (1 - animation.value) * -8),
            child: child,
          ),
        );
      },
      child: _CircleIconButton(icon: icon, onTap: onTap),
    );
  }
}

/// Pill rendered just below the top bar when a collection is active. Mirrors
/// the QR chip's shape so the two can stack without visual conflict.
class _CollectionChip extends StatelessWidget {
  const _CollectionChip({required this.collection, required this.onClear});

  final Collection collection;
  final VoidCallback onClear;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.amberAccent, width: 1.2),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.collections_bookmark,
              size: 14, color: Colors.amberAccent),
          const SizedBox(width: 6),
          Flexible(
            child: Text(
              collection.name,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          const SizedBox(width: 8),
          GestureDetector(
            onTap: onClear,
            child: const Icon(Icons.close, size: 14, color: Colors.white70),
          ),
        ],
      ),
    );
  }
}

/// Pill rendered just below the top bar when a QR sample tag is active.
/// Tapping the X clears it; otherwise it sticks with future captures.
class _QrTagChip extends StatelessWidget {
  const _QrTagChip({required this.metadata, required this.onClear});

  final SampleQrMetadata metadata;
  final VoidCallback onClear;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.lightBlueAccent, width: 1.2),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.qr_code, size: 14, color: Colors.lightBlueAccent),
          const SizedBox(width: 6),
          Flexible(
            child: Text(
              metadata.displayLabel ?? 'QR metadata attached',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          const SizedBox(width: 8),
          GestureDetector(
            onTap: onClear,
            child: const Icon(Icons.close, size: 14, color: Colors.white70),
          ),
        ],
      ),
    );
  }
}

/// Small pill rendered just below the top bar while chatbot mode is on.
/// It complements the shimmer border with an explicit text label so the
/// affordance is unambiguous.
class _ChatbotModeChip extends StatelessWidget {
  const _ChatbotModeChip();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.lightBlueAccent, width: 1.2),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.smart_toy_outlined,
              size: 14, color: Colors.lightBlueAccent),
          SizedBox(width: 6),
          Text(
            'Chatbot mode · capture or import to chat',
            style: TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

class _ModeChip extends StatelessWidget {
  const _ModeChip({required this.mode, required this.onTap});

  final DetectionMode mode;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.white70, width: 1.2),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.local_florist, size: 16, color: Colors.white),
            const SizedBox(width: 6),
            Text(
              mode.label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(width: 4),
            const Icon(Icons.expand_more, size: 16, color: Colors.white70),
          ],
        ),
      ),
    );
  }
}

class _BottomBar extends StatelessWidget {
  const _BottomBar({
    required this.onCapture,
    required this.onPickGallery,
    required this.galleryThumb,
    required this.zoom,
    required this.zoomIndicatorVisible,
    required this.onScanQr,
  });

  final VoidCallback onCapture;
  final VoidCallback onPickGallery;
  final Uint8List? galleryThumb;
  final double zoom;
  final bool zoomIndicatorVisible;
  final VoidCallback onScanQr;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: SizedBox(
        height: 84,
        child: Stack(
          alignment: Alignment.center,
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: _GalleryThumbButton(
                thumb: galleryThumb,
                onTap: onPickGallery,
              ),
            ),
            GestureDetector(
              onTap: onCapture,
              child: Container(
                width: 84,
                height: 84,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 5),
                ),
                child: Container(
                  margin: const EdgeInsets.all(6),
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            ),
            Align(
              alignment: Alignment.centerRight,
              child: _CircleIconButton(
                icon: Icons.qr_code_scanner,
                onTap: onScanQr,
                size: 60,
              ),
            ),
            Align(
              alignment: Alignment.centerRight,
              child: IgnorePointer(
                child: _ZoomIndicator(
                  zoom: zoom,
                  visible: zoomIndicatorVisible,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ZoomIndicator extends StatelessWidget {
  const _ZoomIndicator({required this.zoom, required this.visible});

  final double zoom;
  final bool visible;

  @override
  Widget build(BuildContext context) {
    final label = zoom < 1.0
        ? '${zoom.toStringAsFixed(1)}x'
        : '${zoom.toStringAsFixed(zoom < 10 ? 1 : 0)}x';
    return IgnorePointer(
      ignoring: !visible,
      child: AnimatedOpacity(
        opacity: visible ? 1.0 : 0.0,
        duration: const Duration(milliseconds: 250),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.black54,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white70, width: 1.5),
          ),
          child: Text(
            label,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }
}

class _GalleryThumbButton extends StatelessWidget {
  const _GalleryThumbButton({required this.thumb, required this.onTap});

  final Uint8List? thumb;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 52,
        height: 52,
        clipBehavior: Clip.antiAlias,
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.white70, width: 1.5),
        ),
        child: thumb != null
            ? Image.memory(thumb!, fit: BoxFit.cover, gaplessPlayback: true)
            : const Icon(
                Icons.add_photo_alternate_outlined,
                color: Colors.white70,
                size: 24,
              ),
      ),
    );
  }
}

class _CircleIconButton extends StatelessWidget {
  const _CircleIconButton({
    required this.icon,
    required this.onTap,
    this.highlighted = false,
    this.size = 40,
  });

  final IconData icon;
  final VoidCallback onTap;
  final bool highlighted;
  final double size;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          color: highlighted ? Colors.white : Colors.black54,
          shape: BoxShape.circle,
          border: highlighted
              ? Border.all(color: Colors.lightBlueAccent, width: 1.4)
              : null,
        ),
        child: Icon(
          icon,
          color: highlighted ? Colors.black : Colors.white,
          size: size * 0.5,
        ),
      ),
    );
  }
}

/// Horizontal row of lens-preset chips (e.g. `.5x`, `1x`, `2x`, `5x`). Each
/// chip sets the camera's zoom level to that focal-length multiplier; on
/// iOS this also triggers the OS-level lens swap between physical cameras
/// (ultra-wide, wide, tele). Presets outside the current camera's reported
/// zoom range are filtered out so we never offer an unreachable level.
class _LensPresetBar extends StatelessWidget {
  const _LensPresetBar({
    required this.zoom,
    required this.minZoom,
    required this.maxZoom,
    required this.onSelect,
  });

  final double zoom;
  final double minZoom;
  final double maxZoom;
  final ValueChanged<double> onSelect;

  static const List<double> _candidates = [0.5, 1.0, 2.0, 5.0, 10.0];

  @override
  Widget build(BuildContext context) {
    final presets = _candidates
        .where((p) => p >= minZoom - 0.01 && p <= maxZoom + 0.01)
        .toList();
    // Always offer the camera's true min if it falls below our smallest preset
    // (some sensors report 0.6x rather than 0.5x).
    if (presets.isEmpty || (presets.first - minZoom).abs() > 0.05) {
      if (minZoom < (presets.isEmpty ? 1.0 : presets.first)) {
        presets.insert(0, minZoom);
      }
    }
    if (presets.length < 2) return const SizedBox(height: 0);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          for (final level in presets) ...[
            _LensChip(
              level: level,
              active: (zoom - level).abs() < 0.15,
              onTap: () => onSelect(level),
            ),
            const SizedBox(width: 8),
          ],
        ]..removeLast(),
      ),
    );
  }
}

class _LensChip extends StatelessWidget {
  const _LensChip({
    required this.level,
    required this.active,
    required this.onTap,
  });

  final double level;
  final bool active;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final label = level < 1.0
        ? '.${(level * 10).round()}x'
        : (level == level.roundToDouble()
            ? '${level.toInt()}x'
            : '${level.toStringAsFixed(1)}x');
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: EdgeInsets.symmetric(
          horizontal: active ? 12 : 10,
          vertical: active ? 8 : 6,
        ),
        decoration: BoxDecoration(
          color: active ? Colors.white : Colors.black54,
          shape: BoxShape.rectangle,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: active ? Colors.amberAccent : Colors.white24,
            width: 1.2,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: active ? Colors.black : Colors.white,
            fontSize: active ? 13 : 12,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
    );
  }
}

class _PermissionPrompt extends StatelessWidget {
  const _PermissionPrompt({required this.onRetry});
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: Colors.black,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.no_photography, color: Colors.white54, size: 64),
            const SizedBox(height: 16),
            const Text(
              'Camera permission required',
              style: TextStyle(color: Colors.white, fontSize: 18),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () async {
                if (await Permission.camera.isPermanentlyDenied) {
                  await openAppSettings();
                } else {
                  onRetry();
                }
              },
              child: const Text('Grant access'),
            ),
          ],
        ),
      ),
    );
  }
}
