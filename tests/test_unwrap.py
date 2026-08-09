"""PBC unwrap tests: cross-boundary continuity, known-D recovery, idempotency."""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

from dump_io import unwrap_ortho  # noqa: E402
from conftest import make_traj  # noqa: E402


def _cell_from_box(box):
    n = box.shape[0] if box.ndim == 3 else 1
    cell = np.zeros((n, 3, 3))
    b = box.reshape(n, 3, 2)
    cell[:, 0, 0] = b[:, 0, 1] - b[:, 0, 0]
    cell[:, 1, 1] = b[:, 1, 1] - b[:, 1, 0]
    cell[:, 2, 2] = b[:, 2, 1] - b[:, 2, 0]
    return cell


def test_unwrap_recovers_continuity():
    """After unwrap: (a) every adjacent-frame displacement is < L/2;
    (b) recovered coords equal the true walk modulo an integer number of cells."""
    uw, wrapped, box, _ = make_traj(n_frames=500, n_atoms=50, box_len=25.0, sigma=0.2)
    n = uw.shape[0]
    cell = _cell_from_box(box)
    recovered = unwrap_ortho(wrapped, np.broadcast_to(cell, (n, 3, 3)))
    d = np.diff(recovered, axis=0)
    assert np.all(np.abs(d) < 12.5), "unwrap left a >= L/2 jump somewhere"
    # unwrap is only defined up to an integer cell offset per atom (PBC erases
    # the absolute position); the residue must be an integer multiple of L
    residue = (recovered - uw) % 25.0
    assert np.allclose(np.minimum(residue, 25.0 - residue), 0.0, atol=1e-6)


def test_unwrap_recovers_known_diffusion():
    """A wrapped random walk unwrapped then MSD-fitted must recover sigma^2/dt.

    Large sample (8000 frames x 500 atoms) so the finite-sample MSD estimate
    converges: ratio measured ~1.01 (see tuning notes in git history).
    """
    n_frames, n_atoms, box_len, sigma = 8000, 500, 30.0, 0.2
    uw, wrapped, box, _ = make_traj(n_frames, n_atoms, box_len, sigma=sigma)
    cell = _cell_from_box(box)
    recovered = unwrap_ortho(wrapped, np.broadcast_to(cell, (n_frames, 3, 3)))
    msd = np.mean(np.sum((recovered - recovered[0]) ** 2, axis=2), axis=1)
    t = np.arange(n_frames, dtype=float)
    m = (t >= 0.2 * t[-1]) & (t <= 0.9 * t[-1])
    slope = np.polyfit(t[m], msd[m], 1)[0]
    # MSD = 6 * D * t with D = sigma^2/(2 dt), dt=1 -> slope = 3 sigma^2
    assert slope == pytest.approx(3 * sigma**2, rel=0.05)


def test_unwrap_idempotent_on_unwrapped():
    """unwrap_ortho must be a near-no-op on already-unwrapped input."""
    uw, _, box, _ = make_traj(n_frames=300, n_atoms=30, box_len=20.0, sigma=0.1)
    cell = _cell_from_box(box)
    again = unwrap_ortho(uw, np.broadcast_to(cell, (300, 3, 3)))
    assert np.allclose(again, uw, atol=1e-8)


def test_unwrap_single_frame():
    coords = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    cell = np.array([[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]])
    out = unwrap_ortho(coords, cell)
    assert np.allclose(out, coords)
