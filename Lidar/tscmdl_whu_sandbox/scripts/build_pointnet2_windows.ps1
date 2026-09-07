param(
    [switch]$StrictCompilerCheck
)

$ErrorActionPreference = "Stop"

$Sandbox = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $Sandbox)
$Python = Join-Path $ProjectRoot "envs\whu-tscmdl\python.exe"
$CudaHome = if ($env:CUDA_HOME) {
    $env:CUDA_HOME
} else {
    Join-Path $env:LOCALAPPDATA "TREE-runtime\cuda-12.8"
}
if (-not (Test-Path -LiteralPath (Join-Path $CudaHome "bin\nvcc.exe"))) {
    $CondaCudaHome = Join-Path $CudaHome "Library"
    if (Test-Path -LiteralPath (Join-Path $CondaCudaHome "bin\nvcc.exe")) {
        $CudaHome = $CondaCudaHome
    }
}
$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$BuildScript = Join-Path $PSScriptRoot "build_pointnet2_windows.py"

foreach ($RequiredPath in @($Python, $CudaHome, $VcVars, $BuildScript)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path not found: $RequiredPath"
    }
}

$CommandParts = @(
    "set `"CUDA_HOME=$CudaHome`"",
    "set `"CUDA_PATH=$CudaHome`"",
    "set `"TORCH_CUDA_ARCH_LIST=8.9`"",
    "set `"MAX_JOBS=4`"",
    "set `"DISTUTILS_USE_SDK=1`"",
    "set `"PATH=$CudaHome\bin;$(Split-Path -Parent $Python);$(Join-Path (Split-Path -Parent $Python) 'Scripts');%PATH%`""
)

if (-not $StrictCompilerCheck) {
    $CommandParts += 'set "POINTNET2_ALLOW_UNSUPPORTED_COMPILER=1"'
}

$CommandParts += "call `"$VcVars`""
$CommandParts += "`"$Python`" `"$BuildScript`""
$Command = $CommandParts -join " && "

cmd.exe /d /s /c $Command
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
