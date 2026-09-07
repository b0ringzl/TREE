"""Build PointMLP's pointnet2_ops without changing the installed PyTorch package."""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
POINTNET2_ROOT = (
    SANDBOX_ROOT / "vendor" / "pointMLP-pytorch" / "pointnet2_ops_lib"
)


def main() -> None:
    if not POINTNET2_ROOT.exists():
        raise FileNotFoundError(f"PointNet2 source not found: {POINTNET2_ROOT}")

    import torch.utils.cpp_extension as cpp_extension

    # Localized cl.exe output is not guaranteed to match Python's OEM codec.
    # Replacement is safe here because the compiler version digits are ASCII.
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("MAX_JOBS", "4")
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")

    missing_tools = [name for name in ("cl", "nvcc", "ninja") if not shutil.which(name)]
    if missing_tools:
        raise RuntimeError(f"Build tools are not visible in PATH: {', '.join(missing_tools)}")

    os.chdir(POINTNET2_ROOT)
    sys.argv = [str(POINTNET2_ROOT / "setup.py"), "build_ext", "--inplace"]
    runpy.run_path(str(POINTNET2_ROOT / "setup.py"), run_name="__main__")


if __name__ == "__main__":
    main()
