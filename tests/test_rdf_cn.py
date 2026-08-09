"""l2_rdf_cn integration tests: lattice-pair g(r) peak position and CN count."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _write_system(tmp_path, o_positions, box=20.0, n_frames=5, label="O"):
    """Write a small dump + data: n oxygen atoms held fixed, all type 1 (element O)."""
    n = len(o_positions)
    data = tmp_path / "sys.data"
    data.write_text(
        f"LAMMPS data for rdf test\n\n{n} atoms\n1 atom types\n\n"
        f"0.0 {box} xlo xhi\n0.0 {box} ylo yhi\n0.0 {box} zlo zhi\n\n"
        "Masses\n\n1 15.999 # O\n\nAtoms # full\n\n"
        + "".join(f"{i + 1} 1 1 {x:.4f} {y:.4f} {z:.4f}\n"
                  for i, (x, y, z) in enumerate(o_positions)),
        encoding="utf-8")

    lines = []
    for f in range(n_frames):
        lines += ["ITEM: TIMESTEP", str(f), "ITEM: NUMBER OF ATOMS", str(n),
                  "ITEM: BOX BOUNDS pp pp pp", f"0 {box}", f"0 {box}", f"0 {box}",
                  "ITEM: ATOMS id type x y z"]
        lines += [f"{i + 1} 1 {x:.4f} {y:.4f} {z:.4f}"
                  for i, (x, y, z) in enumerate(o_positions)]
    dump = tmp_path / "sys.dump"
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data, dump


def run_rdf(tmp_path, data, dump, *extra):
    cmd = [sys.executable, str(SCRIPTS / "l2_rdf_cn.py"),
           "--dump", str(dump), "--data", str(data),
           "--outdir", str(tmp_path), "--name", "r", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"rdf_cn failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_rdf_peak_and_cn(tmp_path):
    """Two O atoms 3.0 A apart: g(r) peak at 3.0, CN(<3.5)=1."""
    data, dump = _write_system(tmp_path, [(0, 0, 0), (3.0, 0, 0)])
    r = run_rdf(tmp_path, data, dump, "--pairs", "O:O", "--nbins", "300",
                "--rmin", "0.5", "--rmax", "10", "--cn-cut", "3.5")

    csv = tmp_path / "O-O_rdf.csv"
    assert csv.is_file()
    r_arr = np.loadtxt(csv, delimiter=",", skiprows=1)
    rvals, g, cn = r_arr[:, 0], r_arr[:, 1], r_arr[:, 2]
    i_peak = int(np.argmax(g))
    assert abs(rvals[i_peak] - 3.0) < 0.05, f"peak at {rvals[i_peak]}"
    assert g[i_peak] > 5.0, f"peak height {g[i_peak]} too weak"
    cn_at_cut = cn[np.searchsorted(rvals, 3.5) - 1]
    assert cn_at_cut == np.float64(1.0) or abs(cn_at_cut - 1.0) < 0.3, f"CN={cn_at_cut}"
    assert "CN(<3.5" in r.stdout


def test_rdf_no_peak_reported(tmp_path):
    """Two O atoms far apart (8 A): no peak in [1.5, 5.0] -> reported as such."""
    data, dump = _write_system(tmp_path, [(0, 0, 0), (8.0, 0, 0)])
    r = run_rdf(tmp_path, data, dump, "--pairs", "O:O", "--nbins", "300",
                "--rmin", "0.5", "--rmax", "10", "--cn-cut", "3.5")
    assert "no peak in [1.5,5.0]" in r.stdout


def test_rdf_missing_element_errors(tmp_path):
    data, dump = _write_system(tmp_path, [(0, 0, 0), (3.0, 0, 0)])
    cmd = [sys.executable, str(SCRIPTS / "l2_rdf_cn.py"),
           "--dump", str(dump), "--data", str(data),
           "--pairs", "Na:O", "--outdir", str(tmp_path), "--name", "r"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
    assert "elements-map" in r.stderr or "elements-map" in r.stdout
