"""l1_log_parse tests: variable-column thermo blocks, drift/block equilibrium."""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from l1_log_parse import block_average, drift_check, energy_column, parse_thermo  # noqa: E402


def _write_log(path, blocks):
    """blocks: list of (header_list, rows_list_of_lists)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("LAMMPS (xx) ...\n\n")
        for header, rows in blocks:
            fh.write(" ".join(header) + "\n")
            for row in rows:
                fh.write(" ".join(f"{v:g}" for v in row) + "\n")
            fh.write("\n")


def test_parse_single_block(tmp_path):
    p = tmp_path / "log.lammps"
    _write_log(p, [(["Step", "Temp", "PotEng"], [[0, 300, -100.0], [10, 310, -99.0]])])
    data, avail = parse_thermo(p)
    assert avail == ["Temp", "PotEng"]
    assert data["Step"].tolist() == [0, 10]
    assert data["Temp"].tolist() == [300.0, 310.0]


def test_parse_variable_columns_merged_on_intersection(tmp_path):
    """Two blocks with different extra columns must merge on their intersection."""
    p = tmp_path / "log.lammps"
    _write_log(p, [
        (["Step", "Temp", "PotEng"], [[0, 300, -100.0], [10, 310, -99.0]]),
        (["Step", "Temp", "Press", "Vol"], [[20, 305, 1.0, 1000.0], [30, 308, 1.1, 999.0]]),
    ])
    data, avail = parse_thermo(p)
    # intersection: Step, Temp
    assert avail == ["Temp"]
    assert data["Step"].tolist() == [0, 10, 20, 30]
    assert data["Temp"].tolist() == [300.0, 310.0, 305.0, 308.0]


def test_parse_skips_text_lines(tmp_path):
    p = tmp_path / "log.lammps"
    _write_log(p, [(["Step", "Temp"], [[0, 300.0]])])
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("WARNING: this is a text line with 2 tokens\n")
    data, _ = parse_thermo(p)
    assert len(data["Step"]) == 1


def test_no_thermo_block_raises(tmp_path):
    p = tmp_path / "log.lammps"
    p.write_text("nothing here\n")
    with pytest.raises(ValueError, match="no thermo block"):
        parse_thermo(p)


def test_energy_column_priority():
    assert energy_column(["Press", "PotEng", "Temp"]) == "PotEng"
    assert energy_column(["Press", "Temp"]) is None
    assert energy_column(["TotEng", "PotEng"]) == "TotEng"


def test_drift_check_flat_is_equilibrated():
    # white noise about a constant mean: linear fit slope ~ 0 -> equilibrated
    # (finite-sample slope scatter is ~ sigma*sqrt(12/N) = 0.11 sigma for
    # N=1000, so a tolerance of 0.5 sigma passes it comfortably)
    rng = np.random.default_rng(0)
    t = np.arange(1000, dtype=float)
    y = 100.0 + rng.normal(0.0, 0.1, 1000)
    ok, slope, sigma = drift_check(t, y, tol=0.5)
    assert ok
    assert sigma == pytest.approx(0.1, rel=0.1)


def test_drift_check_ramp_not_equilibrated():
    t = np.arange(1000, dtype=float)
    y = 100.0 + 0.01 * t
    ok, slope, sigma = drift_check(t, y, tol=0.1)
    assert not ok
    assert slope == pytest.approx(0.01)


def test_block_average():
    y = np.arange(100, dtype=float)
    blocks = block_average(y, 5)
    assert len(blocks) == 5
    means = [b[0] for b in blocks]
    assert means[0] == pytest.approx(9.5)
    assert means[-1] == pytest.approx(89.5)
    assert all(b[1] == pytest.approx(np.std(np.arange(20))) for b in blocks)
