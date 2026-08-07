"""E4 full -- (b) A0 support-stratified attenuation curve (spec v0.2 E4b).

Cells: direction jumps {10,25,45}deg x rho* {1.0, 0.5, 0.0} + novel_translation at
the three matched magnitudes, x seeds. For each run the informed A0 tracker gives
m_k; we measure

    recovered fraction  = ||m_K - m_0|| / ||v_K - v_0||     (what the anchors saw)
    realized sqrt(rho)  = ||P_hist (v_K - v_0)|| / ||v_K - v_0||   (what was on-support)

Amendment A3/A4: the prediction is recovered ~ sqrt(rho) (slope 1), plotted against
the realized (logged) sqrt(rho) -- NOT rho, and NOT the nominal rho_target (rotations
have a floor rho = (1-cos theta)/2). NT cells sit at sqrt(rho) = 0 exactly.

CLI: ``python -m graderdrift.experiments.e4b --out DIR [--seeds 0-19] [--nproc N]``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..drift import Event, Schedule
from .e3_run import run_a0_filter

HIST = np.arange(16)
EVENT_T = 30


def _nt_mag(angle_deg):
    return 2.0 * np.sin(np.radians(angle_deg) / 2.0)   # ||delta|| of a rotation at ||v||=1


def e4b_schedules() -> dict[str, Schedule]:
    S: dict[str, Schedule] = {"S0": Schedule("S0", "A0")}   # on-support noise floor
    for ang in (10, 25, 45):
        for rt in (1.0, 0.5, 0.0):
            S[f"rot{ang}_r{int(rt * 10)}"] = Schedule(f"rot{ang}_r{int(rt * 10)}", "A0",
                [Event(EVENT_T, "direction_jump", {"angle_deg": ang, "rho_target": rt})])
    for ang in (10, 25, 45):
        S[f"NT{ang}"] = Schedule(f"NT{ang}", "A0",
            [Event(EVENT_T, "novel_translation", {"magnitude": _nt_mag(ang)})])
    return S


def _one(job):
    out_root, name, seed = job
    sched = e4b_schedules()[name]
    res = run_a0_filter(name, seed, mode="informed", K=60, schedule=sched)
    dm = res["means"][-1] - res["means"][0]              # estimate displacement
    dv = res["v_true_traj"][-1] - res["v0_true"]         # true drift
    ndv = float(np.linalg.norm(dv))
    dv_on = dv.copy(); dv_on[16:] = 0.0                  # P_hist dv (A0 battery span)
    on = float(np.linalg.norm(dv_on))
    # Recovered on-support drift = SIGNED projection of the estimate's displacement
    # onto the true on-support drift direction. This is the noise-unbiased estimator
    # of ||P_A delta|| (magnitude ||dm|| is floor-inflated; noise . uhat is mean 0).
    if on > 1e-9:
        rec = float(dm @ (dv_on / on))
    else:
        rec = float(np.linalg.norm(dm[HIST]))            # NT/S0: on-support truth is 0
    onmag = float(np.linalg.norm(dm[HIST]))              # on-support estimate movement (floor stat)
    payload = {"name": name, "seed": seed,
               "true_drift": ndv, "sqrt_rho": (on / ndv) if ndv > 1e-9 else 0.0,
               "recovered": rec, "recovered_frac": (rec / ndv) if ndv > 1e-9 else 0.0,
               "onsupport_mag": onmag}
    p = Path(out_root) / f"{name}_{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    return (name, seed)


def run(out_root, seeds, nproc=1):
    jobs = [(out_root, name, seed) for name in e4b_schedules() for seed in seeds
            if not (Path(out_root) / f"{name}_{seed}.json").exists()]
    print(f"{len(jobs)} jobs", flush=True)
    if nproc > 1:
        import multiprocessing as mp
        with mp.Pool(nproc) as pool:
            for r in pool.imap_unordered(_one, jobs):
                print("done", r, flush=True)
    else:
        for j in jobs:
            print("done", _one(j), flush=True)


def analyze(out_root, seeds):
    rows = [json.loads((Path(out_root) / f"{name}_{seed}.json").read_text())
            for name in e4b_schedules() for seed in seeds]
    rot = [n for n in e4b_schedules() if n.startswith("rot")]

    # --- rotation cells: recovered_frac vs realized sqrt_rho (the attenuation curve) ---
    print(f"{'cell':11s} {'sqrt_rho':>9s} {'recovered_frac':>15s} {'(pred=sqrt_rho)':>16s}")
    for name in rot:
        rs = [r for r in rows if r["name"] == name]
        sr = np.mean([r["sqrt_rho"] for r in rs]); rf = np.mean([r["recovered_frac"] for r in rs])
        print(f"{name:11s} {sr:>9.3f} {rf:>15.3f} {sr:>16.3f}")

    rot_rows = [r for r in rows if r["name"].startswith("rot")]
    allx = np.array([r["sqrt_rho"] for r in rot_rows])
    ally = np.array([r["recovered_frac"] for r in rot_rows])
    rng = np.random.default_rng(0)
    seeds_arr = np.array(sorted(seeds))
    slopes, intr = [], []
    for _ in range(4000):
        pick = rng.choice(seeds_arr, size=len(seeds_arr), replace=True)
        idx = [i for s_ in pick for i, r in enumerate(rot_rows) if r["seed"] == s_]
        A = np.polyfit(allx[idx], ally[idx], 1)
        slopes.append(A[0]); intr.append(A[1])
    lo, hi = np.percentile(slopes, [5, 95])
    print(f"\nslope(recovered_frac ~ sqrt_rho) 90% CI [{lo:.3f}, {hi:.3f}] "
          f"-> {'PASS (includes 1)' if lo <= 1.0 <= hi else 'CHECK'}; "
          f"intercept {np.mean(intr):+.3f}")

    # --- NT off-support cells vs the S0 on-support noise floor ---
    floor = np.array([r["onsupport_mag"] for r in rows if r["name"] == "S0"])
    fmean, fse = floor.mean(), floor.std(ddof=1) / np.sqrt(len(floor))
    print(f"\nS0 on-support noise floor ||dm_hist|| = {fmean:.3f} +/- {fse:.3f}")
    print(f"{'NT cell':8s} {'||dm_hist||':>11s} {'vs floor+2se':>13s}")
    for ang in (10, 25, 45):
        nt = np.array([r["onsupport_mag"] for r in rows if r["name"] == f"NT{ang}"])
        ok = nt.mean() <= fmean + 2 * floor.std(ddof=1)
        print(f"NT{ang:<6d} {nt.mean():>11.3f} {'within' if ok else 'ABOVE':>13s} "
              f"(off-support drift {'invisible' if ok else 'leaking'})")


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    if not a.report_only:
        run(a.out, seeds, nproc=a.nproc)
    analyze(a.out, seeds)
