param(
    [switch]$StrictCompilerCheck
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:LOCALAPPDATA "miniconda3\envs\yolo\python.exe"
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
$Source = Join-Path $ProjectRoot "Lidar\pctrees_github\PCT_Pytorch\pointnet2_ops_lib"

foreach ($RequiredPath in @($Python, $CudaHome, $VcVars, (Join-Path $Source "setup.py"))) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path not found: $RequiredPath"
    }
}

$PythonRoot = Split-Path -Parent $Python
$CommandParts = @(
    "set `"CUDA_HOME=$CudaHome`"",
    "set `"CUDA_PATH=$CudaHome`"",
    "set `"TORCH_CUDA_ARCH_LIST=8.9`"",
    "set `"MAX_JOBS=4`"",
    "set `"DISTUTILS_USE_SDK=1`"",
    "set `"PATH=$CudaHome\bin;$PythonRoot;$(Join-Path $PythonRoot 'Scripts');%PATH%`""
)
if (-not $StrictCompilerCheck) {
    $CommandParts += 'set "POINTNET2_ALLOW_UNSUPPORTED_COMPILER=1"'
}
$CommandParts += "call `"$VcVars`""
$CommandParts += "cd /d `"$Source`""
$CommandParts += "`"$Python`" setup.py build_ext --inplace"

cmd.exe /d /s /c ($CommandParts -join " && ")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
