# Packaging audit — 2026-10-23

The former 2026-09-20 portable bundle was copied into new versioned locations;
the archived bundle and the production Demo remain unchanged. The new private
package overlays current Identify, Video, VMMS, SAM, world-catalogue and
Dropbox logic and verifies the original deployed model bytes before staging.
Absolute teacher/head paths are made relative **only in staged copies**.

The installed package contains a self-contained .NET desktop host, bundled
Conda runtime, current application code, required models/catalogue/reference
files and the pinned Android download artifacts. The original VMMS panoramas
are not copied; `desktop-settings.json` points to the local source archive.

Validated on this computer: 3,132 application files passed the package
SHA-256 manifest; GPU check passed on RTX 4090 D; the installed window reached
the local page; model health and an actual image prediction passed. In the
working build, SAM segmentation, tree detection, a real 10-second video job
and a short VMMS corridor job completed. A mocked-picker ASGI check confirmed
the Dropbox local-origin gate accepts the package's dynamic port and rejects
an external origin. See `DELIVERY_REPORT.md` for exact evidence.

Remaining release gates: interactive Dropbox/Microsoft account sign-in,
unlocked-window visual acceptance, clean-machine install/upgrade/rollback,
and redistribution-rights review. The source `release.json` deliberately keeps
`portable_windows_current=false`; local success does not establish a public
release or cross-machine compatibility.
