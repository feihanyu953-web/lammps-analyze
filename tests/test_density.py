"""l2_density integration tests: uniform density, layer splitting."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _write_uniform_dump(tmp_path, n=50000, box=10.0, seed=11, zcenters=None):
    rng = np.random.default_rng(seed)
    if zcenters is None:
        pos = rng.uniform(0.0, box, size=(n, 3))
    else:
        # two Gaussian slabs around zcenters (0.3 A spread), uniform xy
        n2 = n // len(zcenters)
        pos = []
        for zc in zcenters:
            z = rng.normal(zc, 0.3, n2)
            xy = rng.uniform(0.0, box, size=(n2, 2))
            pos.append(np.column_stack([xy, z]))
        pos = np.concatenate(pos)[:n]
    dump = tmp_path / "d.dump"
    lines = ["ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", str(n),
             "ITEM: BOX BOUNDS pp pp pp", f"0 {box}", f"0 {box}", f"0 {box}",
             "ITEM: ATOMS id type x y z"]
    lines += [f"{i + 1} 1 {x:.4f} {y:.4f} {z:.4f}" for i, (x, y, z) in enumerate(pos)]
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dump, box, n


def run_density(tmp_path, dump, *extra):
    cmd = [sys.executable, str(SCRIPTS / "l2_density.py"),
           "--dump", str(dump), "--outdir", str(tmp_path), "--name", "d", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"density failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_uniform_density(tmp_path):
    dump, box, n = _write_uniform_dump(tmp_path)
    run_density(tmp_path, dump, "--nbins", "100")
    csv = tmp_path / "d_density.csv"
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    rho = data[:, 1]
    mean_rho = float(rho.mean())
    assert abs(mean_rho - n / box**3) < 0.05 * n / box**3, f"mean rho={mean_rho}"


def test_gaussian_slabs_split_into_layers(tmp_path):
    dump, box, n = _write_uniform_dump(tmp_path, zcenters=[2.0, 8.0])
    run_density(tmp_path, dump, "--nbins", "100", "--layers", "--smooth", "5")
    layers = tmp_path / "d_layers.csv"
    assert layers.is_file()
    data = np.loadtxt(layers, delimiter=",", skiprows=1)
    n_layers = len(data)
    # two Gaussian slabs -> local minimum between them -> 2 or 3 layers
    assert n_layers in (2, 3), f"{n_layers} layers detected"
    # the central split must lie between the slabs (~5 A)
    if n_layers == 2:
        assert abs(data[0, 2] - data[0, 1] - 5.0) < 1.5 or \
               abs(data[1, 1] - 5.0) < 1.5


def test_number_vs_mass_density(tmp_path):
    dump, box, n = _write_uniform_dump(tmp_path, n=20000)
    run_density(tmp_path, dump, "--density-type", "mass", "--nbins", "50",
                "--elements-map", "1:O")
    csv = tmp_path / "d_density.csv"
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    rho_mass = float(data[:, 1].mean())
    # all type 1 = O: mass density = rho_num * 15.999 u/A^3 * 1.66054 g/cm3
    rho_num = n / box**3
    assert abs(rho_mass - rho_num * 15.999 * 1.66054) < 0.1 * rho_mass
