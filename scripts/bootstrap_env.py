#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create and maintain the isolated lammps-analyze runtime environment.

Creates <skill-dir>/.venv with uv (Python 3.11) and installs the analysis
dependencies (numpy / matplotlib / MDAnalysis) into it, so that skill scripts
never pollute or conflict with other Python environments on the machine.

Idempotent: if the venv and its .ready marker exist, it prints "ready" and
exits 0 without doing anything. Use --force to rebuild from scratch, or
--skip-install to create the venv without installing packages (troubleshooting).

Exit codes:
  0  ready (already built) or built successfully
  2  uv not found (installation instructions printed)
  3  dependency installation failed (.ready not written)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = SKILL_DIR / ".venv"
READY_MARKER = VENV_DIR / ".ready"
REQUIREMENTS = SKILL_DIR / "requirements.txt"
PYTHON_VERSION = "3.11"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _uv_available() -> bool:
    return shutil.which("uv") is not None


def _run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"==> {label}")
    print("    " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd)


def is_ready() -> bool:
    return venv_python().is_file() and READY_MARKER.is_file()


def write_ready_marker() -> None:
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uv_version": subprocess.run(
            ["uv", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "python_version": subprocess.run(
            [str(venv_python()), "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "requirements_sha256": _file_hash(REQUIREMENTS) if REQUIREMENTS.is_file() else None,
    }
    READY_MARKER.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"==> wrote {READY_MARKER}")


def build(skip_install: bool = False) -> int:
    if not _uv_available():
        print(
            "ERROR: `uv` not found on PATH.\n"
            "  Install it with:  pip install uv\n"
            "  or on Windows:    winget install astral-sh.uv\n"
            "  Alternative (manual): python -m venv <skill-dir>/.venv --python 3.11\n"
            "    then: <venv>/python -m pip install -r <skill-dir>/requirements.txt\n"
        )
        return 2

    r = _run(["uv", "venv", str(VENV_DIR), "--python", PYTHON_VERSION], "uv venv")
    if r.returncode != 0:
        print(f"ERROR: uv venv failed (exit {r.returncode})")
        return 3

    if not skip_install:
        if not REQUIREMENTS.is_file():
            print(f"ERROR: {REQUIREMENTS} missing")
            return 3
        r = _run(
            [
                "uv", "pip", "install",
                "--python", str(venv_python()),
                "-r", str(REQUIREMENTS),
            ],
            "uv pip install -r requirements.txt",
        )
        if r.returncode != 0:
            print(f"ERROR: dependency install failed (exit {r.returncode}); .ready NOT written")
            return 3

    write_ready_marker()
    print("==> environment ready")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap the lammps-analyze .venv environment")
    ap.add_argument("--force", action="store_true", help="delete and rebuild the .venv")
    ap.add_argument("--skip-install", action="store_true", help="create venv without installing deps")
    args = ap.parse_args()

    if args.force and VENV_DIR.exists():
        print(f"==> removing {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    if is_ready():
        meta = json.loads(READY_MARKER.read_text(encoding="utf-8"))
        print(f"==> ready (created {meta.get('created')}, {meta.get('python_version')})")
        print(f"    {venv_python()}")
        return 0

    if VENV_DIR.exists() and not READY_MARKER.is_file():
        # venv exists but install was interrupted or .ready lost: reinstall into it
        print("==> .venv exists without .ready marker; re-running install")
        r = _run(
            [
                "uv", "pip", "install",
                "--python", str(venv_python()),
                "-r", str(REQUIREMENTS),
            ],
            "uv pip install -r requirements.txt",
        )
        if r.returncode != 0:
            return 3
        write_ready_marker()
        return 0

    return build(skip_install=args.skip_install)


if __name__ == "__main__":
    sys.exit(main())
