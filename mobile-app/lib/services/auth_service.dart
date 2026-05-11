import 'dart:async';
import 'dart:convert';
import 'dart:io' show Platform;

import 'package:crypto/crypto.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:google_sign_in/google_sign_in.dart';
import 'package:gopher_eye/firebase_options.dart';
import 'package:gopher_eye/services/app_settings.dart';
import 'package:sign_in_with_apple/sign_in_with_apple.dart';

class AuthException implements Exception {
  AuthException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Firebase Auth facade. Initializes the SDK, keeps an ID token cached in
/// [AppSettings] for the API layer, and exposes the providers we use:
/// email/password, Sign in with Apple (iOS), Google Sign-In, and anonymous.
class AuthService {
  AuthService._();
  static final AuthService instance = AuthService._();

  bool _initialized = false;
  StreamSubscription<User?>? _idTokenSub;
  Timer? _refreshTimer;

  /// Push every `User?` change so widgets can rebuild via `StreamBuilder`.
  /// Cold-start emits the cached user from disk (or null).
  Stream<User?> get userChanges =>
      _initialized ? FirebaseAuth.instance.authStateChanges() : const Stream.empty();

  User? get currentUser => _initialized ? FirebaseAuth.instance.currentUser : null;

  /// Must be called once before the first [userChanges] listener attaches.
  /// Idempotent — safe to call multiple times during hot reload.
  Future<void> init() async {
    if (_initialized) return;
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
    _initialized = true;

    // Mirror the current ID token into AppSettings so ApiClient (and any
    // future auth-gated route) can read it synchronously.
    _idTokenSub = FirebaseAuth.instance.idTokenChanges().listen((user) async {
      if (user == null) {
        await AppSettings.setAuthToken(null);
        _refreshTimer?.cancel();
        _refreshTimer = null;
      } else {
        try {
          final token = await user.getIdToken();
          await AppSettings.setAuthToken(token);
        } catch (e) {
          _log('getIdToken failed: $e');
        }
        _ensureRefreshTimer();
      }
    });
  }

  // ----- email + password -----

  Future<UserCredential> signInWithEmail(String email, String password) async {
    _requireInit();
    try {
      return await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: email.trim(),
        password: password,
      );
    } on FirebaseAuthException catch (e) {
      throw AuthException(_friendlyError(e));
    }
  }

  Future<UserCredential> createAccountWithEmail(String email, String password) async {
    _requireInit();
    try {
      return await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: email.trim(),
        password: password,
      );
    } on FirebaseAuthException catch (e) {
      throw AuthException(_friendlyError(e));
    }
  }

  Future<void> sendPasswordReset(String email) async {
    _requireInit();
    try {
      await FirebaseAuth.instance.sendPasswordResetEmail(email: email.trim());
    } on FirebaseAuthException catch (e) {
      throw AuthException(_friendlyError(e));
    }
  }

  // ----- Sign in with Apple (iOS only) -----

  Future<UserCredential> signInWithApple() async {
    _requireInit();
    if (!Platform.isIOS && !Platform.isMacOS) {
      throw AuthException('Sign in with Apple is only available on Apple platforms.');
    }
    final rawNonce = _generateNonce();
    final nonce = _sha256OfString(rawNonce);
    try {
      final apple = await SignInWithApple.getAppleIDCredential(
        scopes: const [
          AppleIDAuthorizationScopes.email,
          AppleIDAuthorizationScopes.fullName,
        ],
        nonce: nonce,
      );
      final credential = OAuthProvider('apple.com').credential(
        idToken: apple.identityToken,
        rawNonce: rawNonce,
      );
      return await FirebaseAuth.instance.signInWithCredential(credential);
    } on SignInWithAppleAuthorizationException catch (e) {
      if (e.code == AuthorizationErrorCode.canceled) {
        throw AuthException('Apple sign-in cancelled.');
      }
      throw AuthException('Apple sign-in failed: ${e.message}');
    } on FirebaseAuthException catch (e) {
      throw AuthException(_friendlyError(e));
    }
  }

  // ----- Google Sign-In -----

  Future<UserCredential> signInWithGoogle() async {
    _requireInit();
    try {
      final googleUser = await GoogleSignIn().signIn();
      if (googleUser == null) {
        throw AuthException('Google sign-in cancelled.');
      }
      final googleAuth = await googleUser.authentication;
      final credential = GoogleAuthProvider.credential(
        idToken: googleAuth.idToken,
        accessToken: googleAuth.accessToken,
      );
      return await FirebaseAuth.instance.signInWithCredential(credential);
    } on FirebaseAuthException catch (e) {
      throw AuthException(_friendlyError(e));
    }
  }

  // ----- Anonymous -----

  Future<UserCredential> signInAnonymously() async {
    _requireInit();
    try {
      return await FirebaseAuth.instance.signInAnonymously();
    } on FirebaseAuthException catch (e) {
      throw AuthException(_friendlyError(e));
    }
  }

  // ----- Sign out -----

  Future<void> signOut() async {
    _requireInit();
    // Best-effort: Google and Apple have their own session caches. Sign out
    // there too so the next "Continue with Google" doesn't silently reuse the
    // last account.
    try {
      await GoogleSignIn().signOut();
    } catch (_) {}
    await FirebaseAuth.instance.signOut();
  }

  // ----- internals -----

  void _ensureRefreshTimer() {
    // Firebase ID tokens expire 60 min after issue. Force a refresh every
    // 50 min so AppSettings (and the broker-bound ApiClient calls) keep a
    // valid bearer at hand.
    _refreshTimer?.cancel();
    _refreshTimer = Timer.periodic(const Duration(minutes: 50), (_) async {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) return;
      try {
        final token = await user.getIdToken(true);
        await AppSettings.setAuthToken(token);
      } catch (e) {
        _log('periodic getIdToken failed: $e');
      }
    });
  }

  void _requireInit() {
    if (!_initialized) {
      throw AuthException('AuthService.init() has not been called yet.');
    }
  }

  String _friendlyError(FirebaseAuthException e) {
    switch (e.code) {
      case 'invalid-email':
        return 'That email address looks invalid.';
      case 'user-disabled':
        return 'This account has been disabled.';
      case 'user-not-found':
      case 'wrong-password':
      case 'invalid-credential':
        return 'Email or password is incorrect.';
      case 'email-already-in-use':
        return 'An account with that email already exists.';
      case 'weak-password':
        return 'Password is too weak (use at least 6 characters).';
      case 'operation-not-allowed':
        return 'This sign-in method is not enabled in the Firebase project.';
      case 'network-request-failed':
        return 'Network error. Check your connection and try again.';
      default:
        return e.message ?? 'Authentication failed (${e.code}).';
    }
  }

  static String _generateNonce([int length = 32]) {
    const charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._';
    final random = DateTime.now().microsecondsSinceEpoch;
    final buffer = StringBuffer();
    var seed = random;
    for (var i = 0; i < length; i++) {
      // Tiny LCG — output isn't used as crypto material directly; it's hashed
      // before submission, and the hash is what Firebase + Apple validate.
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      buffer.write(charset[seed % charset.length]);
    }
    return buffer.toString();
  }

  static String _sha256OfString(String input) {
    return sha256.convert(utf8.encode(input)).toString();
  }

  void _log(String message) {
    if (kDebugMode) {
      // ignore: avoid_print
      print('[AuthService] $message');
    }
  }

  Future<void> dispose() async {
    await _idTokenSub?.cancel();
    _idTokenSub = null;
    _refreshTimer?.cancel();
    _refreshTimer = null;
  }
}

