# SpaceScan Trees desktop release and rollback

The public `TREE` repository contains source and release tooling. Complete
Windows packages go only to the access-controlled
`b0ringzl/SpaceScan-Trees-Desktop-Private` GitHub repository. Never attach the
model pack, downloaded reference imagery, local VMMS archive or initialized
installation to a public release.

## Local development checkpoint

Before changing deployed code, run:

```powershell
python desktop_app/release/create_local_snapshot.py `
  --repo C:\path\to\TREE `
  --output-root C:\path\to\TREE\local_snapshots
```

This makes a new timestamped source ZIP, SHA-256 manifest and Git-state record
and verifies the copy. It does not alter the current Git index or working tree.
Keep older snapshots and Git commits. The separate pre-training backup policy
in `AGENTS.md` still applies to **every** training run; a source snapshot is
not a backup of model weights, training data or an installed package.

## Build a private release bundle

Inputs are (1) the never-launched 2026-09-20 portable base, (2) the locally
validated desktop staging package, and (3) a full source commit SHA. The
release builder reads the pristine runtime from the base and only the
checksum-verified application allowlist from the staging package. It strips
machine-specific settings, logs, WebView2 profile and `user_data`, then emits
an immutable versioned directory of sub-2-GiB parts plus a SHA-256 manifest.
It never overwrites an existing output directory.

```powershell
python desktop_app/release/build_release.py `
  --base C:\path\to\SpaceScan-Trees-H120-Windows `
  --stage C:\path\to\SpaceScan-Trees-Desktop-Private-20261023 `
  --output C:\path\to\releases\v2026.10.24.1 `
  --version v2026.10.24.1 `
  --source-snapshot-manifest C:\path\to\local_snapshots\source-...\manifest.json `
  --source-commit FULL_40_CHARACTER_COMMIT_SHA
```

Run `python desktop_app/release/test_release.py -v` before building. Add the
part files, the `.release.json` manifest, the source snapshot and installer
scripts to the same **private** GitHub Release tag. The guarded
`Publish-PrivateRelease.ps1` checks repository privacy and remote SHA-256
before making a draft release downloadable. A GitHub source-code ZIP is not
the desktop installer.

## Install on another Windows PC

Prerequisites: Windows 10/11 x64, a supported NVIDIA GPU/driver, WebView2
Runtime, sufficient free storage (the unpacked app is about 15 GB, and the
installer temporarily needs room for both the compressed archive and the
unpacked version), and a GitHub account with access to the private repository.
For one-command authenticated download, install GitHub CLI and sign in:

```powershell
gh auth login
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Install-FromGitHub.ps1 `
  -Version v2026.10.24.1
```

Alternatively, download **all** assets from the private Release page into one
folder, then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Install-SpaceScanTrees.ps1 `
  -BundleDir C:\path\to\downloaded-release
```

The installer verifies every part before extraction and every extracted file
before activation. It installs to a new version folder, runs the bundled
`launch.py --check-only`, and only then updates desktop/Start Menu shortcuts.
It keeps prior versions and backs up prior shortcuts. If GPU/model validation
fails, shortcuts are not changed. VMMS imagery is selected separately through
the folder-picker; cloud accounts require their own login/configuration.

To return shortcuts to an earlier verified version:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Switch-SpaceScanTreesVersion.ps1 `
  -Version v2026.10.24.1
```

Release status must remain **private evaluation** until an independent PC has
passed install, startup, image/SAM, video, VMMS and cloud-login acceptance.
Check distribution rights and notices for every model, dependency and bundled
reference image before expanding access.
