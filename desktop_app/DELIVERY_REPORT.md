# Private desktop delivery — 2026-10-23

- Installed executable: `%LOCALAPPDATA%\Programs\SpaceScan Trees\20261023\SpaceScan.Trees.Desktop.exe`
- Desktop and Start Menu shortcut: `SpaceScan Trees Desktop`
- Recoverable working build: `desktop_app/dist/SpaceScan-Trees-Desktop-Private-20261023`
- Archived base left intact: `C:\SST_Pack_20260920_v1\SpaceScan-Trees-H120-Windows`
- Source Demo and deployed weights: unchanged; no training performed.

## Verification completed

1. The staged and installed copies each passed `launch.py --check-only`,
   including 3,132 file hashes and the GPU/BF16 check on RTX 4090 D.
2. The installed WPF executable started its **own** teacher and Demo services
   at ports 18339/18340, while the original 8039/8043 Demo remained available.
   `/api/health` reported `ready=true`, aligned H120 image classifier,
   available detector and available world-sequence classifier.
3. Installed image inference on the bundled regression image completed with
   H120 Top-1 `Acacia auriculiformis` (99.6% model score). This smoke score is
   not an accuracy estimate.
4. In the working build: tree detection returned a candidate, SAM 2.1 returned
   a 32-point polygon, a real 10-second video job completed with four tracks,
   and a short VMMS corridor job completed. Video/VMMS used the world model.
5. The installed `/video`, `/corridor`, `/apps`, Android guide and VMMS catalogue
   routes returned HTTP 200; the catalogue found 14 route segments.
6. The local Dropbox origin gate returned 200 for the package's localhost port
   and 403 for an external origin with the native picker mocked. The existing
   Dropbox app key was rendered in the installed Identify page.
7. The desktop shortcut target was checked and the installed executable was
   running. The desktop was locked during screenshot capture, so visual UI
   acceptance and interactive OAuth/login remain unverified.

The VMMS panorama archive, local Dropbox files and new job outputs remain
outside the immutable model manifest. This is an internal/private evaluation
install, not a public distribution package.

## Startup repair — 2026-10-23

The installed desktop shortcut was rechecked and still targets the private WPF
build. A cold start on this computer takes about 36 seconds (roughly 16 seconds
to verify bundled files and 17 seconds to initialize the services). Previously
the window showed an unchanged indeterminate loader during that entire interval.
It now shows the current startup stage and elapsed time, and reports WebView2
initialization/navigation failures rather than waiting indefinitely.

The portable supervisor now waits for a previous shutdown request to finish
before reusing local services. Closing the window during startup terminates only
that window's unfinished supervisor and its children. The installed and staged
copies were updated after backing up the previous launcher, DLL and executable
under `desktop_app/repair_backups/20261023-startup-v1`.

Two launches through the desktop shortcut reached the recognition page and the
window remained responsive. A close-and-immediate-reopen test also succeeded;
closing during startup left no packaged Python process or pending state file.
No model weights, production Demo files or training data changed.

## Faster repeat startup — 2026-10-23

Profiling showed the 3,132-file SHA-256 pass took 13.88 seconds per launch.
The portable launcher now performs a full SHA-256 verification on first use,
when the manifest or any packaged file size/mtime changes, and whenever
`--check-only` is invoked. An unchanged, previously verified installation
uses a quick 0.19-second metadata check. The verification cache is private to
the installation's `user_data` and does not alter any model or source file.

The installed package passed another forced full verification (`CHECK OK`).
Subsequent launch through the desktop shortcut reached the recognition page in
21.7 seconds (22:46:34.9 to 22:46:56.6), down from about 36 seconds. The
remaining time is primarily GPU/model service initialization. The previous
launcher is backed up under `desktop_app/repair_backups/20261023-speed-v1`.

## Icon and startup-screen refresh — 2026-10-24

The user-provided leaf/city icon was preserved and deterministically converted
to a multi-resolution Windows icon. The executable resource, WPF window/taskbar
icon, desktop shortcut and Start Menu shortcut now use it. The loading screen
shows the icon, app name and a short startup status without an elapsed timer.
The prior host and shortcuts are backed up under
`desktop_app/repair_backups/20261023-icon-ui-v1`.

The installed EXE/DLL hashes match the new publish output, Windows extracts a
32-pixel embedded icon from the executable, and launching from the desktop
shortcut on 2026-10-24 reached the recognition page with a responsive window.
The image-recognition Demo, model weights and inference logic were unchanged.

## Transparent rounded icon — 2026-10-24

The previous dark corners were replaced by actual alpha transparency outside
the icon's rounded-square contour. An image edit supplied the contour; the
earlier icon's interior RGB pixels were retained. New versioned PNG/ICO assets
are under `SpaceScan.Trees.Desktop/Assets/SpaceScanTrees-rounded.*`. The EXE,
WPF window/loading page, desktop shortcut and Start Menu shortcut use them.
The prior EXE/DLL and icon are in
`desktop_app/repair_backups/20261024-rounded-icon-v1`.

Verification: the installed EXE hash matches the new publish output; Windows
extracts a 32x32 icon with alpha 0 at the corner; the installed icon matches
the source hash; launch through the desktop shortcut reached the recognition
page with a responsive window. No model, inference, or production Demo change.

## White-background icon — 2026-10-24

The icon was edited to replace the rounded-square's dark-blue interior with a
white interior while retaining the green/cyan leaf and city illustration. The
clean transparent rounded-corner mask from the prior icon was reapplied to
remove generated edge artifacts. The new versioned assets are
`SpaceScan.Trees.Desktop/Assets/SpaceScanTrees-white.png` and `.ico`.
The EXE, WPF window/loading image, desktop shortcut, and Start Menu shortcut
use the white version. Previous binaries, icon and shortcuts are recoverable
under `desktop_app/repair_backups/20261024-white-icon-v1`.

The installed icon matches the source SHA-256, Windows extracts the EXE icon
with a transparent corner, and the desktop shortcut reached the recognition
page with a responsive window on 2026-10-24. Models and inference were not
changed.
