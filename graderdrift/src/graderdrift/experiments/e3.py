"""E3 batch driver -- estimator calibration (spec v0.2 Section 4, E3).

Runs the drift tracker over {S0,S1,S2,S4,S5} x seeds x {known_calendar, blind},
writing one JSONL per run. The analysis (``analysis/e3_analysis.py``) reads only
those logs and produces the coverage table + noise floor + trajectory figure.

S3 (population mixture) is not in this batch: its measurement model is misspecified
(Jensen flattening) so it is reported descriptively, not gated (spec v0.2 E3).

CLI: ``python -m graderdrift.experiments.e3 --out DIR [--seeds 0-19] [--nproc N]``
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .e3_run import ALL_SCHEDULES, SINGLE_TEACHER, run_a0_filter

MODES = ["known_calendar", "blind"]


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _run_one(job):
    out_root, mode, sched, seed, K = job
    run_dir = Path(out_root) / mode / sched / str(seed)
    run_a0_filter(sched, seed, mode=mode, K=K, run_dir=run_dir)
    return (mode, sched, seed)


def run_e3(out_root, seeds, K: int = 60, nproc: int = 1,
           modes: list[str] | None = None, schedules: list[str] | None = None,
           skip_done: bool = True) -> None:
    modes = modes or MODES
    schedules = schedules or SINGLE_TEACHER
    jobs = [(out_root, mode, sched, s, K)
            for mode in modes for sched in schedules for s in seeds]
    if skip_done:  # idempotent resubmission (CHPC pattern)
        jobs = [j for j in jobs
                if not (Path(j[0]) / j[1] / j[2] / str(j[3]) / "log.jsonl").exists()]
    print(f"{len(jobs)} jobs to run", flush=True)
    if nproc > 1:
        import multiprocessing as mp
        with mp.Pool(nproc) as pool:
            for r in pool.imap_unordered(_run_one, jobs):
                print("done", r, flush=True)
    else:
        for j in jobs:
            print("done", _run_one(j), flush=True)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--modes", default=None, help="comma list; default known_calendar,blind")
    ap.add_argument("--schedules", default=None,
                    help=f"comma list; default {','.join(SINGLE_TEACHER)}; all = {','.join(ALL_SCHEDULES)}")
    a = ap.parse_args()
    run_e3(a.out, _parse_seeds(a.seeds), K=a.K, nproc=a.nproc,
           modes=a.modes.split(",") if a.modes else None,
           schedules=a.schedules.split(",") if a.schedules else None)
    print("E3 run complete")
