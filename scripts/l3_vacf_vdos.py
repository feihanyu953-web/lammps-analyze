#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l3: velocity autocorrelation function (VACF) -> vibrational density of
states (VDOS), pure numpy.

Requires velocity columns in the dump ('vx vy vz'; the parser reads them
automatically). VACF(tau) = <v(t0+tau) . v(t0)> / <v(0)^2> averaged over atoms
and time origins; VDOS(nu) = FFT(VACF) (magnitude of the real part), reported
both in THz and cm^-1 (1 THz = 33.356 cm^-1).

Physical notes: the intramolecular O-H stretch appears around 100 THz
(~3600 cm^-1, MACE/AIMD values depend on the potential), librational bands
~10-25 THz. Intermolecular H-bond band ~5-7 THz.

Outputs: <name>_vacf.csv (t_ps, vacf), <name>_vdos.csv (freq_thz, freq_cm,
vdos), <name>_vacf_vdos.png, CONCLUSION line (dominant peak).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import parse_dump, read_types_elements  # noqa: E402
from plot_utils import apply_style, print_conclusion, savefig  # noqa: E402
from region import add_region_args, select_traj  # noqa: E402


def vacf(vel: np.ndarray, tmax: int | None = None) -> np.ndarray:
    """VACF normalized to 1 at tau=0, averaged over atoms and time origins.

    vel: (nframe, natom, 3). Returns VACF over tau = 0..min(tmax, nframe-1).
    """
    nframe, natom, _ = vel.shape
    ntau = min(tmax or nframe, nframe)
    norm = float(np.mean(np.sum(vel * vel, axis=2)))  # <v(0)^2>
    if norm == 0:
        raise ValueError("all velocities are zero; nothing to correlate")
    out = np.zeros(ntau)
    for tau in range(ntau):
        prod = vel[tau:] * vel[: nframe - tau]
        out[tau] = float(np.mean(np.sum(prod, axis=2)))
    return out / norm


def vdos_from_vacf(vacf_arr: np.ndarray, dt_ps: float) -> tuple[np.ndarray, np.ndarray]:
    """VDOS(nu) = Re FFT(VACF); returns (freq_thz, freq_cm1)."""
    n = len(vacf_arr)
    freqs = np.fft.rfftfreq(n, d=dt_ps)  # cycles/ps == THz (1/ps = 1 THz)
    return freqs, freqs * 33.356  # THz -> cm^-1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="l3: VACF -> VDOS (requires vx vy vz columns in the dump)")
    ap.add_argument("--dump", required=True, help="dump.lammpstrj with velocity columns")
    ap.add_argument("--data", default=None, help="data.lammps (type->element)")
    ap.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    ap.add_argument("--stop", type=int, default=-1, help="last frame, exclusive (default all)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--timestep-fs", type=float, default=0.5,
                    help="MD timestep in fs (used when meta/dt inference fails)")
    ap.add_argument("--types", default=None, help="type ids to analyse, e.g. '1 3'")
    ap.add_argument("--elements", default=None,
                    help="element symbols to analyse, e.g. 'O H'")
    ap.add_argument("--tmax", type=int, default=None,
                    help="max lag in frames for VACF (default: full trajectory)")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="vdos", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    d = parse_dump(args.dump, start=args.start, stop=args.stop, stride=args.stride,
                   timestep_fs=args.timestep_fs)
    if d.velocities is None:
        raise SystemExit(
            "ERROR: dump has no velocity columns (vx vy vz). Add them to the "
            "LAMMPS dump command, e.g.: dump ... id type xu yu zu vx vy vz")
    if d.dt_fs is None:
        raise SystemExit("ERROR: cannot infer dt (need >= 2 frames with timestep gaps)")

    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)
    if args.elements:
        tset = set()
        for el in str(args.elements).split():
            hits = [t for t, e in elements.items() if e == el]
            if not hits:
                raise SystemExit(f"ERROR: element '{el}' not in type map {elements}")
            tset.update(hits)
    elif args.types:
        tset = {int(t) for t in str(args.types).split()}
    else:
        raise SystemExit("ERROR: choose atoms with --types or --elements")

    mask = select_traj(d.coords, d.box, args, d.types)[0]
    keep = mask & np.isin(d.types[0], list(tset))
    if not keep.any():
        raise SystemExit("ERROR: selection is empty")
    print(f"    analysing {keep.sum()} atoms, {len(d.timesteps)} frames, "
          f"dt={d.dt_fs} fs")

    vel = d.velocities[:, keep]
    v = vacf(vel, args.tmax)
    dt_ps = d.dt_fs / 1000.0
    freqs_thz, freqs_cm = vdos_from_vacf(v, dt_ps)
    t_ps = np.arange(len(v)) * dt_ps

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv1 = outdir / f"{args.name}_vacf.csv"
    with open(csv1, "w", encoding="utf-8") as fh:
        fh.write("t_ps,vacf\n")
        for i in range(len(v)):
            fh.write(f"{t_ps[i]:.6g},{v[i]:.8g}\n")
    print(f"==> wrote {csv1}")

    spec = np.fft.rfft(v * np.hanning(len(v))).real  # computed here for CSV + plot
    csv2 = outdir / f"{args.name}_vdos.csv"
    with open(csv2, "w", encoding="utf-8") as fh:
        fh.write("freq_thz,freq_cm1,vdos\n")
        for i in range(len(freqs_thz)):
            fh.write(f"{freqs_thz[i]:.6g},{freqs_cm[i]:.6g},{spec[i]:.6g}\n")
    print(f"==> wrote {csv2}")

    apply_style()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    ax1.plot(t_ps, v, lw=1.0)
    ax1.set_xlabel("t (ps)")
    ax1.set_ylabel("VACF")
    ax1.axhline(0, color="gray", lw=0.5)
    ax2.plot(freqs_thz, spec, lw=1.0)
    ax2.set_xlabel("frequency (THz)")
    ax2.set_ylabel("VDOS (a.u.)")
    png = outdir / f"{args.name}_vacf_vdos.png"
    savefig(fig, png)

    i_max = int(np.argmax(spec[1:])) + 1  # skip DC
    print_conclusion(
        f"dominant VDOS peak at {freqs_thz[i_max]:.1f} THz "
        f"({freqs_cm[i_max]:.0f} cm-1), VACF tau-zero={v[0]:.3f}, "
        f"frames={len(v)} dt={d.dt_fs} fs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
