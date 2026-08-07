"""E4a cluster-robust predictor analysis (spec v0.2 E4a; amendments D6, wave 8-9).

Harm and drift are largely SCHEDULE-determined, so run-level (seed-resampled)
statistics are pseudo-replicated (effective N ~= 17, not 340). This module reports
the honest cross-schedule analysis:

- leave-one-SCHEDULE-out CV R^2 for every predictor tier on the same footing;
- schedule-cluster bootstrap CIs for the key increments;
- the schedule-mean-level regression (17 points);
- the tiers:
    H0   anchor drift magnitude                     (observable)
    H1   + norm/direction split                     (observable)
    H2.5 + drift x operational-gap covariance        (observable, NO oracle)
    H2   + signed alignment to the objective w*      (needs w* -> SIM-ONLY)
    H3   true shift/lag decomposition                (oracle ceiling)

H2.5's ``cope`` feature is the anchor-measured drift (m_K - m_0) in the quadratic
form of the round-0 operational-gap covariance C_op -- both quantities observable in
deployment. C_op is recomputed deterministically per seed from the reference stream.
"""
from __future__ import annotations

import numpy as np

from ..envs.grid import get_env
from ..seeding import spawn_generators
from ..experiments.a1_run import _reference_geometry
from ..experiments.e4_full import DRIFT_SCHEDS, _control, _features, _first_event, _load

TIERS = {
    "H0": ["net_disp"],
    "H1": ["net_disp", "net_lognorm", "net_angle"],
    "H2.5 (oracle-free)": ["net_disp", "net_lognorm", "net_angle", "cope"],
    "H2 (needs w*)": ["net_disp", "net_lognorm", "net_angle", "dalign"],
    "H3 (oracle)": ["es", "ee"],
}


def build_rows(root="runs/e4full", seeds=range(20)):
    env = get_env()
    Cop = {}
    for s in seeds:
        gaps, _, _, _, _ = _reference_geometry(env, spawn_generators(s)["stream"])
        Cop[s] = (gaps.T @ gaps) / len(gaps)
    rows = []
    for name in DRIFT_SCHEDS:
        ctrl, fe = _control(name), _first_event(name)
        for s in seeds:
            r = _load(root, name, s)
            c = _load(root, ctrl, s)
            G = float(np.mean(c["gold_wstar"][fe - 1:] - r["gold_wstar"][fe - 1:]))
            nd, nl, na, da = _features(r)
            dh = r["m_k"][-1] - r["m_k"][0]
            cope = float(np.sqrt(max(dh @ Cop[s] @ dh, 0.0)))
            rows.append(dict(name=name, seed=s, G=G, net_disp=nd, net_lognorm=nl,
                             net_angle=na, dalign=da, cope=cope,
                             es=float(np.mean(r["e_shift"][fe - 1:])),
                             ee=float(np.mean(r["e_epi"][fe - 1:]))))
    return rows


def _adj_r2(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
    n, p = len(y), X.shape[1]
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)


def loso_schedule_cv(rows, keys):
    y = np.array([r["G"] for r in rows])
    ys, pr, bs = [], [], []
    for held in DRIFT_SCHEDS:
        tr = [r for r in rows if r["name"] != held]
        te = [r for r in rows if r["name"] == held]
        Xtr = np.array([[r[k] for k in keys] for r in tr]); ytr = np.array([r["G"] for r in tr])
        Xte = np.array([[r[k] for k in keys] for r in te]); yte = np.array([r["G"] for r in te])
        A = np.column_stack([np.ones(len(ytr)), Xtr]); b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        pr += (np.column_stack([np.ones(len(yte)), Xte]) @ b).tolist()
        ys += yte.tolist(); bs += [ytr.mean()] * len(yte)
    ys, pr, bs = map(np.array, (ys, pr, bs))
    return 1 - np.sum((ys - pr) ** 2) / np.sum((ys - bs) ** 2)


def cluster_bootstrap_increment(rows, keysA, keysB, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([r["G"] for r in rows])
    vals = []
    for _ in range(n):
        pick = rng.choice(DRIFT_SCHEDS, len(DRIFT_SCHEDS), replace=True)
        idx = [i for nm in pick for i, r in enumerate(rows) if r["name"] == nm]
        yy = y[idx]
        if np.std(yy) < 1e-9:
            continue
        XA = np.array([[rows[i][k] for k in keysA] for i in idx])
        XB = np.array([[rows[i][k] for k in keysB] for i in idx])
        vals.append(_adj_r2(XA, yy) - _adj_r2(XB, yy))
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def report(root="runs/e4full", seeds=range(20)):
    rows = build_rows(root, seeds)
    y = np.array([r["G"] for r in rows])
    print("leave-one-SCHEDULE-out CV R^2 (same footing) | schedule-mean adj R^2:")
    mrows = [{k: np.mean([r[k] for r in rows if r["name"] == nm])
              for k in ("G", "net_disp", "net_lognorm", "net_angle", "dalign", "cope", "es", "ee")}
             for nm in DRIFT_SCHEDS]
    ym = np.array([r["G"] for r in mrows])
    for name, keys in TIERS.items():
        Xm = np.array([[r[k] for k in keys] for r in mrows])
        print(f"  {name:22s} CV={loso_schedule_cv(rows, keys):+.3f}   sched-mean adjR2={_adj_r2(Xm, ym):+.3f}")
    print("schedule-cluster bootstrap increments (90% CI):")
    print(f"  H2 - H0     {cluster_bootstrap_increment(rows, TIERS['H2 (needs w*)'], TIERS['H0'])}")
    print(f"  H2.5 - H1   {cluster_bootstrap_increment(rows, TIERS['H2.5 (oracle-free)'], TIERS['H1'])}")
    print(f"  H2.5 - H0   {cluster_bootstrap_increment(rows, TIERS['H2.5 (oracle-free)'], TIERS['H0'])}")
    # cost of DROPPING the oracle reference (H2 - H2.5): if CI includes 0, the
    # oracle-free predictor loses no detectable generalization -- the defensible
    # claim (cleaner than a ratio of two small noisy R^2 values).
    print(f"  H2 - H2.5   {cluster_bootstrap_increment(rows, TIERS['H2 (needs w*)'], TIERS['H2.5 (oracle-free)'])}"
          f"   (includes 0 => dropping w* costs no detectable generalization)")
    print(f"  n_schedules={len(DRIFT_SCHEDS)}, seeds={list(seeds)[:1]}..{list(seeds)[-1]}")


if __name__ == "__main__":  # pragma: no cover
    report()
