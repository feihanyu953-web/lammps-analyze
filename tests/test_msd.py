"""l3_msd tests: known-D recovery end-to-end, unit conversion, meta.json use."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _write_walk_dump(tmp_path, n_frames=2000, n_atoms=100, box_len=25.0, sigma=0.15,
                     seed=42, timestep_skip=100):
    """Random walk -> wrapped dump with timestep gaps (timestep_skip per frame)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import make_traj

    uw, wrapped, box, _ = make_traj(n_frames, n_atoms, box_len, sigma=sigma)
    dump = tmp_path / "walk.dump"
    lines = []
    for f in range(n_frames):
        lines += ["ITEM: TIMESTEP", str(f * timestep_skip),
                  "ITEM: NUMBER OF ATOMS", str(n_atoms),
                  "ITEM: BOX BOUNDS pp pp pp", f"0 {box_len}", f"0 {box_len}",
                  f"0 {box_len}", "ITEM: ATOMS id type x y z"]
        lines += [f"{a + 1} 1 {x:.6f} {y:.6f} {z:.6f}"
                  for a, (x, y, z) in enumerate(wrapped[f])]
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dump, box_len, n_atoms, uw


def run_msd(tmp_path, dump, *extra):
    cmd = [sys.executable, str(SCRIPTS / "l3_msd.py"),
           "--dump", str(dump), "--outdir", str(tmp_path), "--name", "m", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"msd failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_known_diffusion_recovered(tmp_path):
    """Unwrap + MSD must recover D = sigma^2/(2 dt) with dt=1 step = 0.5 fs/1000.

    Large sample (6000 frames x 400 atoms) so the finite-sample MSD estimate
    converges (scatter ~ 5%); tolerance 10%.
    """
    n_frames, n_atoms, sigma, ts_skip = 6000, 400, 0.15, 100
    dump, box_len, _, _ = _write_walk_dump(tmp_path, n_frames=n_frames,
                                           n_atoms=n_atoms, sigma=sigma,
                                           timestep_skip=ts_skip)
    r = run_msd(tmp_path, dump, "--unwrap", "pbc", "--types", "1",
                "--timestep-fs", "0.5", "--dims", "3d")
    # expected D: slope_ps = 3*sigma^2 / dt_ps with dt_ps = ts_skip*0.5/1000 = 0.05 ps
    # D = slope_ps*1e-4/6 ; slope_ps = 3*sigma^2/0.05 = 3*0.0225/0.05 = 1.35
    d_expected = 1.35 * 1e-4 / 6.0
    line = next(l for l in r.stdout.splitlines()
                if l.startswith("CONCLUSION: D(all,3d)="))
    d_val = float(line.split("D(all,3d)=")[1].split()[0])
    assert abs(d_val - d_expected) / d_expected < 0.10, f"D={d_val} expected {d_expected}"


def test_xy_vs_3d_divisor(tmp_path):
    """Isotropic walk: D_xy / D_3d = (S_xy/4)/(S_3d/6) with S_xy=(2/3)S_3d -> 1.0."""
    dump, box_len, n_atoms, _ = _write_walk_dump(tmp_path, n_frames=6000,
                                                 n_atoms=400, sigma=0.15)
    r = run_msd(tmp_path, dump, "--unwrap", "pbc", "--types", "1",
                "--timestep-fs", "0.5", "--dims", "all")
    import re
    ds = {m.group(1): float(m.group(2)) for m in
          re.finditer(r"D\(all,(\w+)\)=([\d.e+-]+)", r.stdout)}
    assert "3d" in ds and "xy" in ds
    assert abs(ds["xy"] / ds["3d"] - 1.0) < 0.10


def test_meta_json_used_for_dt(tmp_path):
    """When fed an l1 product (meta.json present), dt_fs must come from meta."""
    n_frames, sigma, ts_skip = 2000, 0.15, 100
    dump, box_len, n_atoms, _ = _write_walk_dump(tmp_path, n_frames=n_frames,
                                                 sigma=sigma, timestep_skip=ts_skip)
    # write a fake meta.json next to the dump (same prefix)
    meta = {"dt_fs": 50.0, "unwrap_mode": "pbc", "types_elements": {"1": "X"}}
    (tmp_path / "walk_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    r = run_msd(tmp_path, dump, "--types", "1")
    assert "meta.json: dt_fs=50.0" in r.stdout


def test_msd_requires_selection(tmp_path):
    dump, _, _, _ = _write_walk_dump(tmp_path, n_frames=50, n_atoms=10)
    cmd = [sys.executable, str(SCRIPTS / "l3_msd.py"), "--dump", str(dump),
           "--outdir", str(tmp_path), "--name", "m"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
    assert "choose atoms" in r.stdout + r.stderr


def test_msd_forbids_per_frame(tmp_path):
    dump, _, _, _ = _write_walk_dump(tmp_path, n_frames=50, n_atoms=10)
    cmd = [sys.executable, str(SCRIPTS / "l3_msd.py"), "--dump", str(dump),
           "--types", "1", "--per-frame", "--outdir", str(tmp_path), "--name", "m"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
    assert "forbids" in r.stdout + r.stderr
