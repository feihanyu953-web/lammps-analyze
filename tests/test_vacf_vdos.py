"""l3_vacf_vdos tests: pure sinusoidal velocity -> VACF oscillation at the
known frequency; CLI end-to-end on a synthetic dump with velocity columns."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from l3_vacf_vdos import vacf, vdos_from_vacf  # noqa: E402


def test_vacf_sinusoid_oscillates_at_known_frequency():
    """v(t) = cos(2 pi f t) with f = 100 THz, dt = 0.5 fs (0.0005 ps)."""
    f_thz = 100.0
    dt_ps = 0.0005
    n = 4096
    t_ps = np.arange(n) * dt_ps
    vel = np.cos(2 * np.pi * f_thz * t_ps)  # (n,) -> reshape (n, 1, 1)
    vel = vel[:, None, None] * np.ones((1, 5, 3))  # 5 atoms, 3 dims
    v = vacf(vel, tmax=n)
    freqs, _ = vdos_from_vacf(v, dt_ps)
    spec = np.fft.rfft(v * np.hanning(n)).real
    i_max = int(np.argmax(spec[1:])) + 1
    assert freqs[i_max] == pytest.approx(f_thz, rel=0.02), \
        f"peak at {freqs[i_max]} THz, expected {f_thz}"
    assert v[0] == pytest.approx(1.0)


def test_vacf_zero_velocities_raises():
    vel = np.zeros((10, 3, 3))
    with pytest.raises(ValueError, match="zero"):
        vacf(vel)


def _write_vel_dump(tmp_path, n_frames=400, n_atoms=8, dt_fs=0.5, seed=7):
    """Random-walk positions + oscillating velocities with known O-H freq."""
    rng = np.random.default_rng(seed)
    box_len = 12.0
    types = np.array([2, 1, 1, 2, 1, 1, 1, 1])[:n_atoms] if n_atoms <= 8 else \
        np.array([2, 1, 1, 2, 1, 1, 1, 1])
    types = types[:n_atoms]
    f_oh = 100.0  # THz
    lines = []
    pos = rng.uniform(0.1, box_len - 0.1, size=(n_atoms, 3))
    for f in range(n_frames):
        t_ps = f * dt_fs / 1000.0
        vel = np.zeros((n_atoms, 3))
        for a in range(n_atoms):
            vel[a] = [np.cos(2 * np.pi * f_oh * t_ps + a),
                      np.sin(2 * np.pi * f_oh * t_ps + a),
                      0.5 * np.cos(2 * np.pi * 20.0 * t_ps)]
        lines += ["ITEM: TIMESTEP", str(f), "ITEM: NUMBER OF ATOMS", str(n_atoms),
                  "ITEM: BOX BOUNDS pp pp pp", f"0 {box_len}", f"0 {box_len}",
                  f"0 {box_len}", "ITEM: ATOMS id type x y z vx vy vz"]
        for a in range(n_atoms):
            x, y, z = pos[a] + 0.1 * f % box_len, pos[a] + 0.0, pos[a] + 0.0
            lines.append(f"{a+1} {types[a]} {pos[a,0]:.4f} {pos[a,1]:.4f} {pos[a,2]:.4f} "
                         f"{vel[a,0]:.5f} {vel[a,1]:.5f} {vel[a,2]:.5f}")
    dump = tmp_path / "v.dump"
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dump


def test_cli_detects_peak(tmp_path):
    dump = _write_vel_dump(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "l3_vacf_vdos.py"),
           "--dump", str(dump), "--types", "1 2", "--timestep-fs", "0.5",
           "--outdir", str(tmp_path), "--name", "v"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"vacf_vdos failed:\n{r.stdout}\n{r.stderr}"
    assert (tmp_path / "v_vacf.csv").is_file()
    assert (tmp_path / "v_vdos.csv").is_file()
    assert "CONCLUSION:" in r.stdout
    # dominant peak ~100 THz (the O-H-like oscillation)
    assert "100" in r.stdout.split("dominant VDOS peak at")[1].split(" ")[0] or \
        any("99." in l or "100." in l for l in r.stdout.splitlines()
            if "dominant VDOS peak" in l)


def test_cli_requires_velocities(tmp_path):
    """A dump without vx vy vz must be rejected with a clear message."""
    content = ("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
               "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
               "ITEM: ATOMS id type x y z\n1 1 0.5 0.5 0.5\n2 2 9.5 9.5 9.5\n")
    p = tmp_path / "novel.dump"
    p.write_text(content, encoding="utf-8")
    cmd = [sys.executable, str(SCRIPTS / "l3_vacf_vdos.py"),
           "--dump", str(p), "--types", "1", "--outdir", str(tmp_path), "--name", "v"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
    assert "velocity" in (r.stdout + r.stderr)
