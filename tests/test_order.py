"""l2_order unit + integration tests: fcc vs random q6, ideal tetrahedron,
LSI on equidistant neighbors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from l2_order import (  # noqa: E402
    lsi,
    min_image,
    neighbor_indices,
    q_steinhardt,
    q_tetrahedral,
)

BOX = np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])
TYPES1 = np.ones(1 + 12, dtype=int)  # all same type for like-atom neighbors


def _nb(coords, center=0, cutoff=1.5):
    types = np.ones(len(coords), dtype=int)
    ids = np.arange(1, len(coords) + 1)
    rows, nb = neighbor_indices(coords, types, ids, [ids[center]], [1], cutoff, BOX)
    return nb


def test_fcc_q6_high_random_q6_low():
    center = np.zeros((1, 3))
    # fcc = 12 nearest neighbours along (+-1,+-1,0) permutations
    fcc12 = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    ], dtype=float) / np.sqrt(2.0)
    coords = np.vstack([center, fcc12])
    nb = _nb(coords)
    q6_fcc = q_steinhardt(coords, [0], nb, 6, BOX)[0]
    # fcc q6 = 0.5745 (Steinhardt 1983); allow numerical slack
    assert q6_fcc == pytest.approx(0.5745, abs=0.01)

    rng = np.random.default_rng(0)
    rand = rng.normal(size=(30, 3))
    rand /= np.linalg.norm(rand, axis=1)[:, None]
    coords2 = np.vstack([center, rand])
    nb2 = _nb(coords2)
    q6_rand = q_steinhardt(coords2, [0], nb2, 6, BOX)[0]
    assert q6_rand < 0.3, f"random q6={q6_rand} should be far below fcc"
    assert q6_rand > 0.05


def test_tetrahedral_order_ideal_vs_flat():
    # ideal tetrahedron vertices (float: in-place division of int arrays casts)
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    tet /= np.linalg.norm(tet, axis=1)[:, None]
    coords = np.vstack([np.zeros((1, 3)), tet])
    nb = _nb(coords)
    q = q_tetrahedral(coords, [0], nb, BOX)[0]
    assert q == pytest.approx(1.0, abs=0.02)

    # four coplanar neighbours (square) -> q_tet = 1 - 3/8 * (4*1/9 + 2*4/9) = 0.5
    flat = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    coords2 = np.vstack([np.zeros((1, 3)), flat])
    nb2 = _nb(coords2)
    q2 = q_tetrahedral(coords2, [0], nb2, BOX)[0]
    assert q2 == pytest.approx(0.5, abs=0.02)


def test_lsi_equidistant_zero():
    oct = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    coords = np.vstack([np.zeros((1, 3)), oct])
    nb = _nb(coords)
    assert lsi(coords, [0], nb, BOX)[0] == pytest.approx(0.0, abs=1e-9)


def test_min_image():
    d = np.array([[19.9, 0.0, 0.0]])  # closer to 0 than to 20 -> -0.1
    assert np.allclose(min_image(d, BOX), [-0.1, 0.0, 0.0])
    d2 = np.array([[9.9, 0.0, 0.0]])  # closer to 0 -> unchanged
    assert np.allclose(min_image(d2, BOX), [9.9, 0.0, 0.0])


def _write_water_dump(tmp_path, n_frames=3, box=12.0):
    """Two isolated waters (all O neighbors > cutoff) -> low order, trivial run."""
    pos = [
        [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
        [6.0, 6.0, 6.0], [6.96, 6.0, 6.0], [5.76, 6.93, 6.0],
    ]
    types = [2, 1, 1, 2, 1, 1]
    data = tmp_path / "w.data"
    data.write_text(
        f"LAMMPS data\n\n6 atoms\n2 atom types\n\n0.0 {box} xlo xhi\n"
        f"0.0 {box} ylo yhi\n0.0 {box} zlo zhi\n\nMasses\n\n"
        "1 1.008 # H\n2 15.999 # O\n\nAtoms # full\n\n"
        + "".join(f"{i+1} 1 {types[i]} {x:.4f} {y:.4f} {z:.4f}\n"
                  for i, (x, y, z) in enumerate(pos)),
        encoding="utf-8")
    lines = []
    for f in range(n_frames):
        lines += ["ITEM: TIMESTEP", str(f), "ITEM: NUMBER OF ATOMS", "6",
                  "ITEM: BOX BOUNDS pp pp pp", f"0 {box}", f"0 {box}", f"0 {box}",
                  "ITEM: ATOMS id type x y z"]
        lines += [f"{i+1} {types[i]} {x:.4f} {y:.4f} {z:.4f}"
                  for i, (x, y, z) in enumerate(pos)]
    dump = tmp_path / "w.dump"
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data, dump


def test_cli_runs_and_outputs(tmp_path):
    data, dump = _write_water_dump(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "l2_order.py"),
           "--dump", str(dump), "--data", str(data), "--elements", "O",
           "--cutoff", "3.5", "--lsi", "--outdir", str(tmp_path), "--name", "o"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"l2_order failed:\n{r.stdout}\n{r.stderr}"
    assert (tmp_path / "o_order.csv").is_file()
    assert (tmp_path / "o_order.png").is_file()
    assert "CONCLUSION:" in r.stdout
    # two isolated O atoms: no neighbors -> q6 = 0
    assert "q6=0.000" in r.stdout
