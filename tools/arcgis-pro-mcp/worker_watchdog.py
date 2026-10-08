"""Kill an ArcPy worker tree if its MCP server parent disappears."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path


SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
WAIT_TIMEOUT = 0x00000102


def process_exists(process_id: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(
        SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        process_id,
    )
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def kill_tree(process_id: int, system_root: Path) -> None:
    taskkill = system_root / "System32" / "taskkill.exe"
    subprocess.run(
        [str(taskkill), "/PID", str(process_id), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: worker_watchdog.py SERVER_PID WORKER_PID SYSTEM_ROOT")
    server_pid = int(sys.argv[1])
    worker_pid = int(sys.argv[2])
    system_root = Path(sys.argv[3])
    while process_exists(server_pid) and process_exists(worker_pid):
        time.sleep(0.5)
    if not process_exists(server_pid) and process_exists(worker_pid):
        kill_tree(worker_pid, system_root)


if __name__ == "__main__":
    main()

