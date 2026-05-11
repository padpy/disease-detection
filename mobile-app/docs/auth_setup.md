# Firebase Auth setup

The app gates `HomeShell` behind sign-in via [`lib/services/auth_service.dart`](../lib/services/auth_service.dart) and [`lib/screens/sign_in_screen.dart`](../lib/screens/sign_in_screen.dart). Four providers are wired up: **email/password, Sign in with Apple, Google Sign-In, and anonymous**. Before the app can build, the steps below need to happen once per developer / project.

## 1. Generate `firebase_options.dart`

The repo ships a stub at `lib/firebase_options.dart` that throws on init. Replace it with the real one:

```bash
# Install the CLI once.
dart pub global activate flutterfire_cli

# From mobile-app/, regenerate against your team's Firebase project.
flutterfire configure
```

Pick the existing iOS + Android targets when prompted; allow the CLI to overwrite `lib/firebase_options.dart`. The file is gitignored.

## 2. Enable each provider in the Firebase Console

`Firebase Console → Authentication → Sign-in method`. Enable all that you intend to use:

- **Email/Password** — toggle on.
- **Anonymous** — toggle on.
- **Apple** — toggle on. For Sign in with Apple you do **not** need a Services ID for the iOS-only flow, but you must enter the Apple Developer Team ID and Key ID/Private Key if you also want web/Android. iOS-only requires nothing beyond the toggle.
- **Google** — toggle on. Use the auto-suggested web client.

After enabling Google, re-download `GoogleService-Info.plist` and `google-services.json` (the console will offer them) so the iOS/Android client IDs match what the SDK expects. Place them in:

- `mobile-app/ios/Runner/GoogleService-Info.plist`
- `mobile-app/android/app/google-services.json`

Both files are gitignored.

## 3. iOS — Sign in with Apple capability

Xcode → Runner target → Signing & Capabilities → `+ Capability` → **Sign in with Apple**. The entitlement file (`ios/Runner/Runner.entitlements`) already includes `com.apple.developer.applesignin`; adding the capability in the Xcode UI mirrors that into the App ID at Apple Developer portal.

## 4. iOS — Google Sign-In URL scheme

Open `ios/Runner/GoogleService-Info.plist` and find the `REVERSED_CLIENT_ID` value (looks like `com.googleusercontent.apps.123456789-abcdefg`). Paste it into `ios/Runner/Info.plist`, replacing the placeholder `REVERSED_CLIENT_ID` string inside the `CFBundleURLTypes` array.

If Google Sign-In is disabled, you can remove that whole `CFBundleURLTypes` block from `Info.plist`.

## 5. Android — Google Sign-In

`google-services.json` (from step 2) is enough — `google_sign_in_android` reads client IDs from it. No further config required as long as `android/build.gradle` already applies the Google Services plugin (it does in the existing project).

## 6. Run

```bash
flutter pub get
cd ios && pod install && cd ..
flutter run
```

On a clean install the app opens to `SignInScreen`. After any successful sign-in (or anonymous tap), the auth state listener in `AuthService` writes the current Firebase ID token into `AppSettings`, and the `_AuthGate` in [`lib/main.dart`](../lib/main.dart) switches to `HomeShell`.

## How the token reaches the server

1. `AuthService.idTokenChanges` listener writes the new ID token to `AppSettings.setAuthToken`.
2. A 50-minute periodic timer in `AuthService` calls `user.getIdToken(true)` so the cached token stays valid (Firebase tokens expire after 60 min).
3. `ApiClient._authHeaders()` reads the token from `AppSettings` and attaches it to `/vpn/config` calls as `Authorization: Bearer <token>`.
4. The server (`server/app/vpn_routes.py`) verifies the token via `firebase_admin.auth.verify_id_token`, derives `peer_id = "user:<uid>"`, and calls the ZeroTier broker to authorize the device's node id on the managed network.

Anonymous accounts behave identically — they still receive an ID token, and the broker assigns them a peer. Their `uid` simply isn't tied to an email. If you later want to link an anonymous session to a real provider, see `FirebaseAuth.instance.currentUser?.linkWithCredential(...)` — out of scope for v1.

## Troubleshooting

- **"firebase_options.dart has not been generated yet"** at launch — you skipped step 1.
- **`PlatformException(sign_in_failed, ...)` on Google** — the URL scheme in `Info.plist` doesn't match `REVERSED_CLIENT_ID`, or the iOS client ID in `GoogleService-Info.plist` is stale.
- **Apple sign-in returns immediately with `canceled`** — capability not enabled in Xcode, or the App ID at Apple Developer portal doesn't have "Sign In with Apple" turned on.
- **`/vpn/config` returns 401** — the server can't verify the token. Confirm `GOOGLE_APPLICATION_CREDENTIALS` points at the same Firebase service-account JSON on the server (`server/.envrc`).
