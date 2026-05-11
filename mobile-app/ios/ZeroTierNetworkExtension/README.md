# ZeroTier Packet Tunnel Provider (iOS)

This directory holds the source for the Packet Tunnel Provider extension that
implements the on-device ZeroTier data plane via libzt. The Runner target
talks to this extension via `NETunnelProviderManager` (see
`Runner/VpnController.swift`).

Source files in this directory are **not yet wired into `Runner.xcodeproj`** —
adding a target is an Xcode UI action that this scaffolding cannot perform
automatically. Follow the steps below in `Runner.xcworkspace`.

## One-time Xcode setup

1. **Add the libzt package** (via SwiftPM)
   - `File ▸ Add Packages…`
   - URL: `https://github.com/zerotier/libzt`. Pick the latest tag.
   - Add `libzt` to the new extension target you create in step 2 (the Runner
     target itself does not need libzt — only the extension does, because
     only the extension owns the packet flow).

2. **Create the extension target**
   - `File ▸ New ▸ Target…`
   - iOS ▸ Network Extension ▸ Packet Tunnel Provider.
   - Product name: `ZeroTierNetworkExtension`
   - Bundle id: `io.npcomplete.gophereye.PacketTunnel` (must match the
     `kTunnelProviderBundleId` constant in `Runner/VpnController.swift`).
   - Xcode generates a stub `PacketTunnelProvider.swift` and `Info.plist`.
     **Delete those generated stubs** and add the files in this directory
     instead:
     - `PacketTunnelProvider.swift`
     - `Info.plist`
     - `ZeroTierNetworkExtension.entitlements`
   - Add `libzt` as a package dependency of the new target.

3. **Move/replace entitlements files**
   - The `Runner.entitlements` and `ZeroTierNetworkExtension.entitlements`
     files in this scaffold need to be referenced as the *Code Signing
     Entitlements* build setting for their respective targets.
     - Runner target ▸ Build Settings ▸ `Code Signing Entitlements` =
       `Runner/Runner.entitlements`.
     - ZeroTierNetworkExtension target ▸ same setting =
       `ZeroTierNetworkExtension/ZeroTierNetworkExtension.entitlements`.

4. **Capabilities** (Signing & Capabilities tab, for *both* targets):
   - Network Extensions ▸ Packet Tunnel.
   - App Groups: `group.io.npcomplete.gophereye.vpn`
   - Keychain Sharing: `io.npcomplete.gophereye.shared`

5. **Apple Developer portal**
   - Both App IDs (Runner and the new extension) need *Network Extensions*,
     *App Groups*, and *Keychain Sharing* enabled. Regenerate provisioning
     profiles after enabling.
   - Submit the
     [Network Extension entitlement request](https://developer.apple.com/contact/request/)
     to Apple. Required for App Store and external TestFlight (does **not**
     block local development or internal TestFlight). Turnaround is typically
     1–3 weeks.

6. **Bundle identifier discipline**
   - Replace `io.npcomplete.gophereye` everywhere if your team prefix differs.
     Touchpoints:
     - `Runner/VpnController.swift` (`kTunnelProviderBundleId`, `kAppGroupId`)
     - `Runner.entitlements`
     - `ZeroTierNetworkExtension.entitlements`
     - the new target's bundle id in Xcode.
     - the App Group identifier.

7. **Wire libzt into `ZeroTierRuntime`** in `PacketTunnelProvider.swift`. The
   stub in this file calls completion immediately with success; replace the
   bodies of `start(networkId:completion:)` and `stop(completion:)` with the
   real libzt calls:

   ```swift
   // Point libzt at the App Group container so the host app can read
   // identity.public out of it.
   let container = FileManager.default.containerURL(
       forSecurityApplicationGroupIdentifier: "group.io.npcomplete.gophereye.vpn"
   )!.appendingPathComponent("zerotier")
   try? FileManager.default.createDirectory(at: container, withIntermediateDirectories: true)
   zts_init_from_storage(container.path)
   zts_node_start()
   zts_net_join(UInt64(networkId, radix: 16)!)
   ```

   Hook the `NEPacketTunnelFlow` to libzt's virtual interface using libzt's
   ethertap (or the ZeroTier-iOS sample app's bridge code) so packets cross
   between the tunnel and ZeroTier's virtual NIC.

## Verifying the wiring

After the steps above, build + run on a **physical iPhone** signed into a
paid developer team:

- App boots without crashing → channels registered.
- In the Flutter Settings screen, flip the new "VPN (ZeroTier)" toggle.
- iOS shows the system "Allow VPN configuration" prompt (first time only).
- The status line below the toggle should transition
  `Fetching config → Installing profile → Connecting → Connected as
  10.66.66.X`.
- Confirm on the broker host that the device's node id appears under
  `zerotier-cli listnetworks` (and is authorized in the controller).

The iOS Simulator **cannot run Packet Tunnel extensions**. Test on device only.

## Files in this directory

| File | Purpose |
|------|---------|
| `PacketTunnelProvider.swift` | `NEPacketTunnelProvider` subclass. Hands packet I/O to libzt. |
| `Info.plist` | Declares the extension point and principal class. |
| `ZeroTierNetworkExtension.entitlements` | Network Extension, App Group, Keychain Sharing. |

The companion `Runner/VpnController.swift` lives in the main app target and
drives the extension via `NETunnelProviderManager`.
