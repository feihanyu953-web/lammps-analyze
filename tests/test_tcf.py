"""l3_tcf unit + integration tests: residence TCF semantics, dipole C(0)=1,
CLI end-to-end with a synthetic two-water trajectory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from l3_tcf import dipole_tcf, fit_tau, residence_tcf  # noqa: E402


def test_residence_static_all_in():
    in_region = np.ones((5, 3), dtype=bool)
    ct, n_ever = residence_tcf(in_region)
    assert n_ever == 3
    # never-leaving molecules: ensemble-averaged C(t) == 1 for all t
    assert np.allclose(ct, 1.0)


def test_residence_leaving_molecule():
    # mol 0 leaves at frame 2, mol 1 stays, mol 2 never enters
    in_region = np.array([
        [True, True, False],
        [True, True, False],
        [False, True, False],
        [False, True, False],
    ])
    ct, n_ever = residence_tcf(in_region)
    assert n_ever == 2
    # t=0: 1 (normalized)
    # t=1: origins t0=0 (both in, both survive), t0=1 (mol0 in frame1 then
    #      leaves -> fails, mol1 ok), t0=2 (mol1 ok) -> (2+1+1)/(2+2+1) = 0.8
    # t=2: t0=0 (mol1 ok), t0=1 (mol1 ok) -> 2/4 = 0.5
    # t=3: t0=0 (mol1 ok) -> 1/2 = 0.5
    assert np.allclose(ct, [1.0, 0.8, 0.5, 0.5])


def test_residence_none_raises():
    with pytest.raises(ValueError, match="ever enters"):
        residence_tcf(np.zeros((5, 3), dtype=bool))


def test_dipole_constant_direction():
    mu = np.tile(np.array([1.0, 0.0, 0.0]), (6, 3, 1))  # constant dipoles
    ct = dipole_tcf(mu)
    assert np.allclose(ct, 1.0)


def test_dipole_rotating_90deg():
    """Dipole rotating 90 deg per frame: C(0)=1, C(1)=0, C(2)=-1."""
    n = 6
    mu = np.zeros((n, 2, 3))
    for f in range(n):
        ang = f * np.pi / 2
        mu[f, :, 0] = np.cos(ang)
        mu[f, :, 1] = np.sin(ang)
    ct = dipole_tcf(mu)
    assert ct[0] == pytest.approx(1.0)
    assert ct[1] == pytest.approx(0.0, abs=1e-9)
    assert ct[2] == pytest.approx(-1.0, abs=1e-9)


def test_fit_tau_exponential():
    t = np.arange(100, dtype=float)
    ct = np.exp(-t / 25.0)
    assert fit_tau(t, ct)["tau"] == pytest.approx(25.0, rel=0.05)


def _write_water_traj(tmp_path, n_frames=10, box=12.0):
    """Two waters: mol1 near z=10 (top-trunc region), mol2 at z=6 (bulk)."""
    base = np.array([
        [1.0, 1.0, 10.0], [1.96, 1.0, 10.0], [0.76, 1.93, 10.0],  # water 1 (O H H)
        [6.0, 6.0, 6.0], [6.96, 6.0, 6.0], [5.76, 6.93, 6.0],     # water 2 (O H H)
    ])
    types = [2, 1, 1, 2, 1, 1]
    data = tmp_path / "w.data"
    data.write_text(
        f"LAMMPS data\n\n6 atoms\n2 atom types\n\n0.0 {box} xlo xhi\n"
        f"0.0 {box} ylo yhi\n0.0 {box} zlo zhi\n\nMasses\n\n"
        "1 1.008 # H\n2 15.999 # O\n\nAtoms # full\n\n"
        + "".join(f"{i+1} 1 {types[i]} {x:.4f} {y:.4f} {z:.4f}\n"
                  for i, (x, y, z) in enumerate(base)),
        encoding="utf-8")
    lines = []
    for f in range(n_frames):
        shift = 0.0 if f < n_frames // 2 else -3.0  # water 1 leaves the layer
        pos = base.copy()
        pos[:3, 2] += shift  # move water 1 in z
        lines += ["ITEM: TIMESTEP", str(f), "ITEM: NUMBER OF ATOMS", "6",
                  "ITEM: BOX BOUNDS pp pp pp", f"0 {box}", f"0 {box}", f"0 {box}",
                  "ITEM: ATOMS id type xu yu zu"]
        lines += [f"{i+1} {types[i]} {x:.4f} {y:.4f} {z:.4f}"
                  for i, (x, y, z) in enumerate(pos)]
    dump = tmp_path / "w.dump"
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data, dump


def test_cli_residence_and_dipole(tmp_path):
    data, dump = _write_water_traj(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "l3_tcf.py"),
           "--dump", str(dump), "--data", str(data),
           "--mode", "both",
           "--region", "layer", "--layer", "top", "--trunc", "2.0",
           "--outdir", str(tmp_path), "--name", "t"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"tcf failed:\n{r.stdout}\n{r.stderr}"
    assert (tmp_path / "t_tcf.csv").is_file()
    assert (tmp_path / "t_tcf.png").is_file()
    assert "CONCLUSION:" in r.stdout
    assert "tau(residence)" in r.stdout and "tau(dipole)" in r.stdout


def test_cli_residence_requires_region(tmp_path):
    data, dump = _write_water_traj(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "l3_tcf.py"),
           "--dump", str(dump), "--data", str(data),
           "--mode", "residence", "--outdir", str(tmp_path), "--name", "t"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
    assert "needs a region" in (r.stdout + r.stderr)


def test_cli_wrapped_dump_rejected(tmp_path):
    # rewrite the dump with wrapped coords (x y z columns) -> must be rejected
    data, dump = _write_water_traj(tmp_path)
    content = dump.read_text(encoding="utf-8").replace("xu yu zu", "x y z")
    dump.write_text(content, encoding="utf-8")
    cmd = [sys.executable, str(SCRIPTS / "l3_tcf.py"),
           "--dump", str(dump), "--data", str(data),
           "--mode", "dipole", "--outdir", str(tmp_path), "--name", "t"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
    assert "unwrapped" in (r.stdout + r.stderr)
