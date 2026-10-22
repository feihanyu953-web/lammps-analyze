"""plot_utils.smooth edge behavior: edge-shrinking windows, no zero-pad dips.

Regression for the lammps2 KCl analysis (2026-10-22): the old
np.convolve(..., 'same') implementation divided by the full window at the
sequence boundaries while zero-padding the missing points, faking
rising/falling ramps at both edges of stationary time series.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from plot_utils import smooth  # noqa: E402


def test_smooth_constant_stays_constant():
    y = np.full(50, 2.5)
    out = smooth(y, 51)
    assert out.shape == y.shape
    assert np.allclose(out, 2.5)


def test_smooth_edges_shrink_not_zero_pad():
    y = np.array([1.0, 2.0, 3.0])
    out = smooth(y, 3)
    # centered shrinking windows: [mean(1,2), mean(1,2,3), mean(2,3)]
    assert np.allclose(out, [1.5, 2.0, 2.5])
    # the old zero-padded convolution gave [1.0, 2.0, 5/3] -- edge dips


def test_smooth_no_edge_dip_on_stationary_series():
    rng = np.random.default_rng(0)
    y = 2.0 + rng.normal(0.0, 0.5, size=200)
    out = smooth(y, 51)
    # smoothed edge values must sit inside the noise band around the true
    # mean, never collapse toward zero the way zero-padded convolution does
    assert out[0] > 1.0
    assert out[-1] > 1.0


def test_smooth_preserves_length_and_window_one_identity():
    y = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert smooth(y, 1).tolist() == y.tolist()
    assert smooth(y, 7).shape == y.shape
