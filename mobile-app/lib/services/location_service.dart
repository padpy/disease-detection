import 'package:geolocator/geolocator.dart';

class LocationService {
  const LocationService._();

  static Future<Position?> tryGetCurrent({
    Duration timeout = const Duration(seconds: 5),
  }) async {
    try {
      if (!await Geolocator.isLocationServiceEnabled()) return null;

      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        return null;
      }

      return await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        timeLimit: timeout,
      );
    } catch (_) {
      return null;
    }
  }
}
