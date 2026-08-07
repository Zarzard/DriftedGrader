"""E4a model-based harm predictor (review round 4).

The linear readout predictors (H0-H2.5) fail to extrapolate to an unseen drift type
(leave-one-type-out CV <= 0) because harm is threshold-nonlinear and the readouts are
a lossy summary. The oracle H3 generalizes (0.91) because the shift/lag decomposition
is the right functional FORM. This predictor **computes** that form instead of
fitting it, using NO oracle -- only the anchor-estimated grader state m_k, the round-0
estimate m_0 as the (deployable) reference, and a policy solver over the achievable
set (the env model):

    MB = mean_{k>=first_event} [ J_{m_0}(pi*_{m_0}) - J_{m_0}(pi*_{m_k}) ]

i.e. how much worse the policy that is optimal for the *currently estimated* grader
would score under the *initial estimated* standard -- a predicted shift, computed via
soft VI, requiring a world model but not the true objective w*.

Reproducible; parallel over runs. CPU-only; run via SLURM (~13k soft-VI solves).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..envs.grid import GAMMA, get_env
from ..experiments.e4_full import DRIFT_SCHEDS, _control, _first_event, _load
from ..policy import evaluate_policy, reward_from_w, soft_value_iteration

_ENV = get_env()
SWEEPS = 60


def _mb_one(args):
    name, seed, root = args
    r = _load(root, name, seed)
    c = _load(root, _control(name), seed)
    fe = _first_event(name)
    P, F = _ENV.P, _ENV.F
    M = r["m_k"]
    m0 = M[0]
    rF = lambda w: reward_from_w(F, w)
    base = evaluate_policy(P, rF(m0), GAMMA, soft_value_iteration(P, rF(m0), GAMMA, sweeps=SWEEPS))
    shifts = [base - evaluate_policy(P, rF(m0), GAMMA,
                                     soft_value_iteration(P, rF(M[k]), GAMMA, sweeps=SWEEPS))
              for k in range(fe - 1, len(M))]
    G = float(np.mean(c["gold_wstar"][fe - 1:] - r["gold_wstar"][fe - 1:]))
    return name, seed, float(np.mean(shifts)), G


def compute(root="runs/e4full", seeds=range(20), nproc=1, out="runs/e4_mb.json"):
    jobs = [(nm, s, root) for nm in DRIFT_SCHEDS for s in seeds]
    if nproc > 1:
        import multiprocessing as mp
        with mp.Pool(nproc) as pool:
            res = pool.map(_mb_one, jobs)
    else:
        res = [_mb_one(j) for j in jobs]
    data = [{"name": n, "seed": s, "mb": mb, "G": g} for n, s, mb, g in res]
    Path(out).write_text(json.dumps(data))
    return data


def _adj_r2(x, y):
    A = np.column_stack([np.ones(len(y)), x]); b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b; r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
    return 1 - (1 - r2) * (len(y) - 1) / max(len(y) - x.shape[1] - 1, 1)


def report(path="runs/e4_mb.json"):
    data = json.loads(Path(path).read_text())
    y = np.array([d["G"] for d in data]); mb = np.array([d["mb"] for d in data])
    names = [d["name"] for d in data]
    # leave-one-type-out CV
    ys, pr, bs = [], [], []
    for held in DRIFT_SCHEDS:
        tr = [i for i, n in enumerate(names) if n != held]
        te = [i for i, n in enumerate(names) if n == held]
        A = np.column_stack([np.ones(len(tr)), mb[tr]]); b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pr += (np.column_stack([np.ones(len(te)), mb[te]]) @ b).tolist()
        ys += y[te].tolist(); bs += [y[tr].mean()] * len(te)
    ys, pr, bs = map(np.array, (ys, pr, bs))
    cv = 1 - np.sum((ys - pr) ** 2) / np.sum((ys - bs) ** 2)
    ra = np.argsort(np.argsort(mb)); rb = np.argsort(np.argsort(y))
    sp = np.corrcoef(ra, rb)[0, 1]
    mrows = [(np.mean([d["G"] for d in data if d["name"] == nm]),
              np.mean([d["mb"] for d in data if d["name"] == nm])) for nm in DRIFT_SCHEDS]
    gm = np.array([a for a, _ in mrows]); mm = np.array([b for _, b in mrows])
    print(f"H_MB (model-based, oracle-free): in-sample adjR2={_adj_r2(mb[:, None], y):.3f}  "
          f"leave-one-TYPE-out CV={cv:+.3f}  Spearman={sp:.3f}  schedule-mean adjR2={_adj_r2(mm[:, None], gm):.3f}")
    print("(compare: linear H0-H2.5 CV all <= 0; oracle H3 CV = 0.91)")


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--nproc", type=int, default=1)
    a = ap.parse_args()
    if a.compute:
        compute(nproc=a.nproc)
    report()
