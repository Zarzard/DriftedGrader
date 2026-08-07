"""E1 driver -- the stationarity null (spec v0.2 Section 4, E1).

Runs A0/S0 across seeds, writing one JSONL log per seed. The analysis
(``analysis/e1_analysis.py``) reads only those logs. A1/S0 x10 is deferred to M6
(needs the C8 gridworld); this driver covers the A0 rung.

CLI (for SLURM): ``python -m graderdrift.experiments.e1 --out DIR --seeds 0-19 [--K 60] [--nproc N]``
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .a0_run import run_a0


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _run_one(args):
    out_root, seed, K = args
    run_dir = Path(out_root) / "S0" / str(seed)
    run_a0("S0", seed, K=K, run_dir=run_dir)
    return seed


def run_e1(out_root: str | Path, seeds: list[int], K: int = 60, nproc: int = 1) -> None:
    out_root = Path(out_root)
    jobs = [(out_root, s, K) for s in seeds]
    if nproc > 1:
        import multiprocessing as mp

        with mp.Pool(nproc) as pool:
            for s in pool.imap_unordered(_run_one, jobs):
                print(f"done seed {s}", flush=True)
    else:
        for j in jobs:
            print(f"done seed {_run_one(j)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--nproc", type=int, default=1)
    a = ap.parse_args()
    run_e1(a.out, _parse_seeds(a.seeds), K=a.K, nproc=a.nproc)
    print("E1 run complete")
