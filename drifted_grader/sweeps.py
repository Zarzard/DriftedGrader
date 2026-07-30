"""Cartesian sweeps, parallel over runs. Writes one parquet per (config, seed).

DRIFT-REPLAY dependency: a REPLAY run needs the {w_t} trajectory of the matched
SAT run at the same (seed, lambda, beta, ...). run_one generates that twin
internally (agent_stream=0, record) then replays it to a fresh agent
(agent_stream=1), so a REPLAY config is self-contained.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import replace
from typing import List, Tuple

from .runner import Config, run, write_parquet, config_hash


RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "runs")


def run_one(cfg: Config, seed: int, runs_root: str = RUNS_ROOT) -> str:
    if cfg.drift.upper() == "REPLAY":
        sat_twin = replace(cfg, drift="SAT", agent_stream=0)
        sat_rows = run(sat_twin, seed, record_w_traj=True)
        w_traj = sat_rows[0]["_w_traj"]
        replay_cfg = replace(cfg, agent_stream=1)
        rows = run(replay_cfg, seed, replay_w_traj=w_traj)
    else:
        rows = run(cfg, seed)
    path = os.path.join(runs_root, cfg.stage, cfg.hash(), f"{seed}.parquet")
    write_parquet(rows, path, cfg.schema)
    return path


def _worker(args):
    cfg, seed, runs_root = args
    try:
        return run_one(cfg, seed, runs_root), None
    except Exception as e:  # X4: crashes counted + reported, never silently dropped
        return None, f"{cfg.hash()} seed={seed}: {type(e).__name__}: {e}"


def run_grid(configs: List[Config], seeds: List[int], n_workers: int = None,
             runs_root: str = RUNS_ROOT) -> Tuple[List[str], List[str]]:
    """Run every (config, seed). Returns (written_paths, crash_reports)."""
    jobs = [(cfg, s, runs_root) for cfg in configs for s in seeds]
    paths, crashes = [], []
    if n_workers and n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            for path, err in pool.imap_unordered(_worker, jobs, chunksize=4):
                (paths if path else crashes).append(path or err)
    else:
        for job in jobs:
            path, err = _worker(job)
            (paths if path else crashes).append(path or err)
    return paths, crashes


def grid_configs(base: Config, **axes) -> List[Config]:
    """Cartesian product of keyword axes applied on top of `base`.
    e.g. grid_configs(base, drift=['SAT','NONE'], lam=[0.1,0.4])."""
    keys = list(axes.keys())
    out = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        out.append(replace(base, **dict(zip(keys, combo))))
    return out
