import 'dart:async';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gopher_eye/screens/home_shell.dart';
import 'package:gopher_eye/screens/sign_in_screen.dart';
import 'package:gopher_eye/services/app_settings.dart';
import 'package:gopher_eye/services/auth_service.dart';
import 'package:gopher_eye/services/vpn_service.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle.light);
  // Prime SharedPreferences so the disk-load + parse runs in the background
  // while the first frame paints. Without this, the very first call (e.g. the
  // user tapping Settings) blocks for ~100-300 ms on the cold load.
  unawaited(SharedPreferences.getInstance());

  // Firebase Auth must initialize before any screen reads the user; we await
  // it before runApp so the auth gate doesn't flash the sign-in screen on
  // every cold start while Firebase rehydrates the cached session.
  runZonedGuarded(() async {
    await AuthService.instance.init();
    unawaited(_bootVpn());
    runApp(const GopherEyeApp());
  }, (error, stack) {
    // Last-ditch — if Firebase init throws (e.g. firebase_options.dart not
    // generated), still surface a usable UI explaining the situation.
    debugPrint('Fatal init error: $error\n$stack');
    runApp(_BootErrorApp(message: '$error'));
  });
}

/// Subscribe to native VPN status, and if the user previously enabled the
/// tunnel, kick off auto-reconnect in the background. Failures are intentional
/// non-fatal — the app remains usable even if the tunnel never comes up.
Future<void> _bootVpn() async {
  await VpnService.instance.init();
  if (await AppSettings.getVpnEnabled()) {
    try {
      await VpnService.instance.enable();
    } catch (_) {
      // Surface in Settings UI; do not block startup.
    }
  }
}

class GopherEyeApp extends StatelessWidget {
  const GopherEyeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gopher Eye',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: Colors.black,
        useMaterial3: true,
        colorScheme: const ColorScheme.dark(
          primary: Colors.white,
          surface: Colors.black,
        ),
      ),
      home: const _AuthGate(),
    );
  }
}

/// Gates the app on Firebase Auth state. Until the user signs in (any
/// provider, including anonymous), [SignInScreen] is shown; once a user
/// exists, [HomeShell] takes over.
class _AuthGate extends StatelessWidget {
  const _AuthGate();

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: AuthService.instance.userChanges,
      initialData: AuthService.instance.currentUser,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting &&
            !snapshot.hasData) {
          return const _BootSplash();
        }
        final user = snapshot.data;
        if (user == null) return const SignInScreen();
        return const HomeShell();
      },
    );
  }
}

class _BootSplash extends StatelessWidget {
  const _BootSplash();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: Colors.black,
      body: Center(child: CircularProgressIndicator(color: Colors.white)),
    );
  }
}

class _BootErrorApp extends StatelessWidget {
  const _BootErrorApp({required this.message});
  final String message;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        backgroundColor: Colors.black,
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Icon(Icons.warning_amber_outlined,
                    color: Colors.amberAccent, size: 48),
                const SizedBox(height: 16),
                const Text(
                  'Gopher Eye could not start',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  message,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white70, fontSize: 13),
                ),
                const SizedBox(height: 16),
                const Text(
                  'If this is a fresh checkout, run `flutterfire configure` '
                  'from mobile-app/ to generate lib/firebase_options.dart, '
                  'then re-run the app.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white54, fontSize: 12),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
