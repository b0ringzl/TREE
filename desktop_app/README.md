# SpaceScan Trees private Windows desktop app

The 2026-10-24 build is a self-contained WPF/WebView2 window plus an isolated
Python inference runtime. It starts its own pair of free localhost ports and
stops only its own services when the window closes. It does not replace or stop
the development Demo on ports 8039/8043, and it does not train or alter weights.

The tested private installation is at
`%LOCALAPPDATA%\Programs\SpaceScan Trees\20261023`.
Open `SpaceScan.Trees.Desktop.exe` there, or use the separate desktop/Start Menu
shortcut named **SpaceScan Trees Desktop**. The working build remains under
`desktop_app/dist/SpaceScan-Trees-Desktop-Private-20261023` for rollback.
The desktop and Start Menu shortcuts, application window and executable share
the white-background, rounded leaf/city icon. The startup screen has no elapsed timer.
The startup screen now shows verification and model-loading stages; a cold
launch on the validated machine now takes roughly 22 seconds after the first
full integrity check; the first launch or a changed package can take longer.
If startup fails,
the window displays an error and the log path instead of spinning indefinitely.

The installed `desktop-settings.json` points `vmms_root` to the current local
VMMS archive. The folder-picker mode can read user-selected VMMS folders without
copying panoramas into the app. The settings file can also hold a Dropbox
Chooser app key or OneDrive client ID; an existing `SPACESCAN_DROPBOX_APP_KEY`
environment variable is inherited. Cloud login still needs network access.

The first launch repairs paths in the bundled Conda runtime and takes longer
than subsequent launches. Do not move an initialized install directory; create
a fresh versioned installation from the clean archived runtime instead.
Application logs are under `%LOCALAPPDATA%\SpaceScan\Trees\logs`, and private
service logs/jobs are under the installation's `user_data` and `app` folders.

This is a **private, locally tested installation**, not a publicly
redistributable release. Before deployment to another machine, check model,
dependency and image redistribution rights and repeat the full clean-machine
test with a supported NVIDIA GPU and WebView2 runtime. Interactive cloud login
and Windows visual QA were not completed on the locked desktop.

Build inputs and verification are recorded in `DELIVERY_REPORT.md`.

The GitHub directory contains source, final icon assets and verification notes
only. It intentionally excludes private installation files, bundled runtimes,
model weights, local settings, VMMS imagery and recoverable repair backups.
