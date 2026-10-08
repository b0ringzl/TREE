$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot 'outputs/android_toolchain_20260918_v1/paths.json'
if (!(Test-Path -LiteralPath $configPath)) { throw 'Run tools/setup_android_build_20260918.py first, or build with Android Studio.' }
$androidTools = Get-Content -LiteralPath $configPath | ConvertFrom-Json
$env:JAVA_HOME = $androidTools.jdk
$env:ANDROID_HOME = $androidTools.sdk
$env:ANDROID_SDK_ROOT = $androidTools.sdk
$env:GRADLE_USER_HOME = Join-Path $projectRoot 'outputs/android_toolchain_20260918_v1/gradle_cache'
# This host's Unix-domain socket implementation rejects local connects.
# A nonexistent socket directory makes the JDK use its supported TCP loopback fallback.
$env:JAVA_TOOL_OPTIONS = '-Djava.net.preferIPv4Stack=true -Djdk.net.unixdomain.tmpdir=C:/nonexistent-spacescan-unix-sockets'
& $androidTools.gradle -p $PSScriptRoot assembleDebug --no-daemon --console=plain
exit $LASTEXITCODE
