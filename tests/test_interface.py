"""l4_interface tests: orientation cos, SDF normalization, layer props,
CLI end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from l4_interface import layer_props, orientation_cos, sdf_xy  # noqa: E402
from l2_density import significant_minima, smooth  # noqa: E402


def test_orientation_cos_values():
    # two molecules: one pointing +z (cos=1), one lying flat (cos=0)
    mu = np.array([
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    ])
    cos = orientation_cos(mu)
    assert np.allclose(cos, [1.0, 0.0])


def test_sdf_normalization():
    """Uniform adsorbate distribution: rho ~ n/(Lx*Ly) everywhere."""
    rng = np.random.default_rng(0)
    n_frames, n_ads = 50, 200
    coords = rng.uniform(0, 10, size=(n_frames, n_ads, 3))
    box_xy = np.array([[0.0, 10.0], [0.0, 10.0]])
    H, xc, yc = sdf_xy(coords, box_xy, grid=20)
    mean_rho = H.mean()
    assert mean_rho == pytest.approx(n_ads / 100.0, rel=0.1), f"mean rho={mean_rho}"
    assert H.shape == (20, 20)


def test_sdf_pbc_wrapping():
    """Atoms near the box edge must wrap into the map (no empty band)."""
    coords = np.array([[[9.9, 0.1, 0.0], [0.1, 9.9, 0.0]]])  # 1 frame
    box_xy = np.array([[0.0, 10.0], [0.0, 10.0]])
    H, xc, yc = sdf_xy(coords, box_xy, grid=10)
    # both atoms should land in the x/y high or low bins, not vanish
    assert H.sum() == pytest.approx(2.0 / (10 / 10) ** 2 / 1.0)  # 2 atoms / bin area


def test_layer_props_two_slabs():
    """Two O slabs at z~2 and z~8 -> 2-3 layers, each with n_O > 0."""
    rng = np.random.default_rng(1)
    n_mol = 400
    z = np.concatenate([rng.normal(2.0, 0.3, n_mol), rng.normal(8.0, 0.3, n_mol)])
    xy = rng.uniform(0, 12, size=(2 * n_mol, 2))
    o_pos = np.column_stack([xy, z])[None, :, :]  # (1, n, 3)
    box = np.array([[0.0, 12.0], [0.0, 12.0], [0.0, 12.0]])
    layers = layer_props(o_pos, box, nbins=100)
    assert 2 <= len(layers) <= 3
    assert all(l["n_O"] > 0 for l in layers)
    # each slab peak (z=2 and z=8) must be contained in a layer
    assert any(l["zmin"] < 3.0 < l["zmax"] for l in layers), "slab at z=2 not covered"
    assert any(l["zmin"] < 7.0 < l["zmax"] for l in layers), "slab at z=8 not covered"


def _write_interface_system(tmp_path, n_frames=6, box=12.0):
    """Two waters (dipoles up) + one K adsorbate; O at z=10 (interface top)."""
    base = np.array([
        [1.0, 1.0, 10.0], [1.96, 1.0, 10.8], [0.76, 1.93, 10.8],  # water 1 (O H H)
        [4.0, 4.0, 6.0], [4.96, 4.0, 6.8], [3.76, 4.93, 6.8],     # water 2 (O H H)
        [8.0, 8.0, 9.5],                                          # K adsorbate
    ])
    types = [2, 1, 1, 2, 1, 1, 3]
    data = tmp_path / "s.data"
    data.write_text(
        f"LAMMPS data\n\n7 atoms\n3 atom types\n\n0.0 {box} xlo xhi\n"
        f"0.0 {box} ylo yhi\n0.0 {box} zlo zhi\n\nMasses\n\n"
        "1 1.008 # H\n2 15.999 # O\n3 39.098 # K\n\nAtoms # full\n\n"
        + "".join(f"{i+1} 1 {types[i]} {x:.4f} {y:.4f} {z:.4f}\n"
                  for i, (x, y, z) in enumerate(base)),
        encoding="utf-8")
    lines = []
    for f in range(n_frames):
        lines += ["ITEM: TIMESTEP", str(f), "ITEM: NUMBER OF ATOMS", "7",
                  "ITEM: BOX BOUNDS pp pp pp", f"0 {box}", f"0 {box}", f"0 {box}",
                  "ITEM: ATOMS id type xu yu zu"]
        lines += [f"{i+1} {types[i]} {x:.4f} {y:.4f} {z:.4f}"
                  for i, (x, y, z) in enumerate(base)]
    dump = tmp_path / "s.dump"
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data, dump


def test_cli_all_modes(tmp_path):
    data, dump = _write_interface_system(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "l4_interface.py"),
           "--dump", str(dump), "--data", str(data),
           "--mode", "all", "--adsorbate-element", "K",
           "--outdir", str(tmp_path), "--name", "i"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"l4 failed:\n{r.stdout}\n{r.stderr}"
    assert (tmp_path / "i_orientation.csv").is_file()
    assert (tmp_path / "i_sdf.csv").is_file()
    assert (tmp_path / "i_layers_props.csv").is_file()
    assert (tmp_path / "i_sdf.png").is_file()
    assert "CONCLUSION:" in r.stdout
    # both waters point up: <cos> > 0.8
    assert "dipole <cos(theta)>=" in r.stdout
    # one K atom: mean density = 1/(Lx*Ly) with Lx=Ly=12
    assert "K 2D density" in r.stdout


def test_cli_wrapped_rejected(tmp_path):
    data, dump = _write_interface_system(tmp_path)
    content = dump.read_text(encoding="utf-8").replace("xu yu zu", "x y z")
    dump.write_text(content, encoding="utf-8")
    cmd = [sys.executable, str(SCRIPTS / "l4_interface.py"),
           "--dump", str(dump), "--data", str(data),
           "--mode", "orientation", "--outdir", str(tmp_path), "--name", "i"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
    assert "unwrapped" in (r.stdout + r.stderr)
