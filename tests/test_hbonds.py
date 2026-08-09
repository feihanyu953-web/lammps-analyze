"""l2_hbonds integration tests: classic two-water geometry -> exactly 1 bond;
static structure -> C(t) == 1. Also tests the C(t)/lifetime unit functions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _two_waters():
    """Two water molecules with one O-H...O H-bond (d=3.2 A, angle=155 deg).

    O1=(0,0,0); H1b bridges toward O2=(3.2,0,0) at 155 deg:
    H = (x, y) with |H-O1|=0.96, angle O1-H-O2 = 155 deg -> x=0.914, y=0.294.
    H1a is placed away (angle to O2 = 103 deg < 150 -> no bond).
    Water 2's own H atoms are placed away from O1.
    """
    O1 = (0.0, 0.0, 0.0)
    O2 = (3.2, 0.0, 0.0)
    H1b = (0.914, 0.294, 0.0)   # bridges to O2 at 155 deg
    H1a = (0.48, -0.831, 0.0)   # angle O1-H1a-O2 = 103 deg
    H2a = (4.16, 0.0, 0.0)      # covalent on O2, away from O1
    H2b = (2.40, 0.93, 0.0)     # angle O2-H2b-O1 = 115 deg
    return [O1, H1a, H1b, O2, H2a, H2b]  # types: O H H O H H


def _write_system(tmp_path, positions, n_frames=5, box=12.0):
    atoms = _two_waters() if positions is None else positions
    n = len(atoms)
    types = [2, 1, 1, 2, 1, 1]  # O H H O H H
    data = tmp_path / "hb.data"
    data.write_text(
        f"LAMMPS data for hbond test\n\n{n} atoms\n2 atom types\n\n"
        f"0.0 {box} xlo xhi\n0.0 {box} ylo yhi\n0.0 {box} zlo zhi\n\n"
        "Masses\n\n1 1.008 # H\n2 15.999 # O\n\nAtoms # full\n\n"
        # Atoms # full columns: ID mol type x y z
        + "".join(f"{i + 1} 1 {types[i]} {x:.4f} {y:.4f} {z:.4f}\n"
                  for i, (x, y, z) in enumerate(atoms)),
        encoding="utf-8")

    lines = []
    for f in range(n_frames):
        lines += ["ITEM: TIMESTEP", str(f), "ITEM: NUMBER OF ATOMS", str(n),
                  "ITEM: BOX BOUNDS pp pp pp", f"0 {box}", f"0 {box}", f"0 {box}",
                  "ITEM: ATOMS id type x y z"]
        lines += [f"{i + 1} {types[i]} {x:.4f} {y:.4f} {z:.4f}"
                  for i, (x, y, z) in enumerate(atoms)]
    dump = tmp_path / "hb.dump"
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data, dump


def run_hbonds(tmp_path, data, dump, *extra):
    cmd = [sys.executable, str(SCRIPTS / "l2_hbonds.py"),
           "--dump", str(dump), "--data", str(data),
           "--outdir", str(tmp_path), "--name", "h", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"hbonds failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_two_waters_one_bond(tmp_path):
    data, dump = _write_system(tmp_path, None)
    r = run_hbonds(tmp_path, data, dump)
    csv = tmp_path / "h_hbonds.csv"
    counts = np.loadtxt(csv, delimiter=",", skiprows=1)[:, 1]
    assert np.all(counts == 1), f"expected 1 bond per frame, got {counts}"
    assert "0.50 per donor" in r.stdout  # 1 bond / 2 donors


def test_ct_decays_linearly_for_static_bond(tmp_path):
    """Static geometry: one bond of run length L=5 -> C(t) = (L-t)/L, i.e.
    [1, 0.8, 0.6, 0.4, 0.2] (continuous autocorrelation definition)."""
    data, dump = _write_system(tmp_path, None)
    run_hbonds(tmp_path, data, dump)
    ct = np.loadtxt(tmp_path / "h_hbond_ct.csv", delimiter=",", skiprows=1)[:, 1]
    assert np.allclose(ct, [1.0, 0.8, 0.6, 0.4, 0.2])
    assert np.all(np.diff(ct) <= 0)  # monotone decay


def test_no_bonds_when_geometry_fails(tmp_path):
    """O2 moved to 6 A: no H-bond within 3.5 A -> zero bonds, tau n/a."""
    atoms = _two_waters()
    o1 = atoms[0]
    h1b = atoms[2]
    # translate everything so O2 sits 6 A from O1 along +x
    atoms = list(atoms)
    shift = 6.0 - 3.2
    atoms[3] = (atoms[3][0] + shift, 0.0, 0.0)
    atoms[4] = (atoms[4][0] + shift, 0.0, 0.0)
    atoms[5] = (atoms[5][0] + shift, 0.0, 0.0)
    data, dump = _write_system(tmp_path, atoms)
    r = run_hbonds(tmp_path, data, dump)
    counts = np.loadtxt(tmp_path / "h_hbonds.csv", delimiter=",", skiprows=1)[:, 1]
    assert np.all(counts == 0)
    assert "tau_hb=n/a" in r.stdout


def test_extract_runs():
    sys.path.insert(0, str(SCRIPTS))
    from l2_hbonds import extract_runs
    assert extract_runs([0, 1, 2, 5, 6]) == [3, 2]
    assert extract_runs([3]) == [1]


def test_compute_ct_static_vs_interrupted():
    sys.path.insert(0, str(SCRIPTS))
    from l2_hbonds import compute_ct
    # one bond present frames 0..4 (static) -> C(t) = 1 - t/5 staircase
    ct, n_runs = compute_ct({(0, 3): [0, 1, 2, 3, 4]}, 5)
    assert n_runs == 1
    assert np.allclose(ct, [1.0, 0.8, 0.6, 0.4, 0.2])
    # two bonds, one interrupted -> average decay
    ct2, n2 = compute_ct({(0, 3): [0, 1, 2], (1, 4): [0, 1, 2]}, 3)
    assert n2 == 2
    assert np.allclose(ct2, [1.0, 2 / 3, 1 / 3])


def test_fit_lifetime_single_exp():
    sys.path.insert(0, str(SCRIPTS))
    from l2_hbonds import fit_lifetime
    t = np.arange(100, dtype=float)
    ct = np.exp(-t / 20.0)
    fit = fit_lifetime(t, ct, double=False)
    assert fit["method"] == "single"
    assert fit["tau"] == pytest.approx(20.0, rel=0.05)


def test_fit_lifetime_flat_returns_none():
    sys.path.insert(0, str(SCRIPTS))
    from l2_hbonds import fit_lifetime
    t = np.arange(10, dtype=float)
    fit = fit_lifetime(t, np.ones(10), double=False)
    assert fit["tau"] is None
