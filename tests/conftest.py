"""Synthetic data factories for lammps-analyze tests (generated, not committed).

Random-walk trajectories with known diffusion coefficient D are used to
verify that MSD analysis recovers D; PBC wrapping at box edges tests unwrap.
"""

from __future__ import annotations

import numpy as np
import pytest


def make_walk(coords_uw, box, rng, dt=0.5, sigma_per_dim=0.3):
    """Wrap an unwrapped random walk into a periodic box.

    coords_uw: (nframe, natom, 3) unwrapped cartesian.
    Returns wrapped coords via modulo of the box (same-size box assumed).
    """
    L = np.diag(box)
    return coords_uw % L


def random_walk_unwrapped(n_frames, n_atoms, box, rng, sigma):
    """Generate a free 3D random walk (unwrapped). Returns (nframe, natom, 3)."""
    steps = rng.normal(0.0, sigma, size=(n_frames, n_atoms, 3))
    return np.cumsum(steps, axis=0)


def make_traj(n_frames=2000, n_atoms=100, box_len=25.0, seed=42, sigma=0.15):
    """Full synthetic trajectory: unwrapped walk -> wrapped coords + box.

    D_expected = sigma^2 / (2 * dt) per dimension (dt = 1 in step units).
    """
    rng = np.random.default_rng(seed)
    box = np.array([[0.0, box_len], [0.0, box_len], [0.0, box_len]])
    uw = random_walk_unwrapped(n_frames, n_atoms, box, rng, sigma)
    wrapped = uw % box_len
    return uw, wrapped, box, rng


@pytest.fixture
def mini_dump_path():
    from pathlib import Path
    return Path(__file__).parent / "fixtures" / "mini.dump"


@pytest.fixture
def mini_data_path():
    from pathlib import Path
    return Path(__file__).parent / "fixtures" / "mini.data"
