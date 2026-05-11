from __future__ import annotations

import platform
import subprocess
import sys


def run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        return completed.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"No disponible: {exc}"


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"OS: {platform.platform()}")
    print("nvidia-smi:")
    print(run(["nvidia-smi"]))


if __name__ == "__main__":
    main()
