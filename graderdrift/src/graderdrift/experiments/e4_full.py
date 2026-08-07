"""E4 full -- (a) A1 harm hierarchy across the schedule/magnitude grid (spec v0.2 E4a).

17 drift schedules + 2 controls, each x seeds, informed tracker inline:
  {S1,S2,S5} x 3 magnitudes (9) + S7a,S7b (2) + S8t,S8m,S8b (3) + SC1-3 (3).
Controls: S0 (all but S7*), S7null (S7 arms). Outcome Y = G_auc (CRN-paired,
frozen reference); Y' = R_auc. Predictor hierarchy H0..H3 (amendment C4 net stats).

Acceptance (amendment D-gate, lead-approved qualitative form): H2 adds significant
incremental adj-R^2 over H0 (bootstrap CI of the increment excludes 0) AND the
named contrasts hold -- S7a < 0 < S7b with |net_disp_a - net_disp_b| CI ~ 0
(equal magnitude, opposite harm), and the S8 ordering (measured, not assumed).

CLI: ``python -m graderdrift.experiments.e4_full --out DIR [--seeds 0-19] [--nproc N]``
     ``--report_only`` to (re)analyze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..drift import Event, Schedule, get_preset_a1
from ..envs.grid import TRUE_W
from .a1_run import run_a1
from .e4_pilot import _adj_r2, _angle, _features


def e4a_schedules() -> dict[str, Schedule]:
    S: dict[str, Schedule] = {"S0": Schedule("S0", "A1"), "S7null": get_preset_a1("S7null")}
    for ang in (10, 25, 45):
        S[f"S1_{ang}"] = Schedule(f"S1_{ang}", "A1",
            [Event(20, "direction_jump", {"angle_deg": ang, "rho_target": 1.0})])
    for fac in (0.995, 0.985, 0.97):
        S[f"S2_{int(round(1000 * fac))}"] = Schedule(f"S2_{int(round(1000 * fac))}", "A1",
            [], {"norm_decay": {"from": 20, "to": None, "factor": fac}})
    for cos in (0.9, 0.7, 0.5):
        S[f"S5_{int(round(100 * cos))}"] = Schedule(f"S5_{int(round(100 * cos))}", "A1",
            [Event(20, "family_swap", {"cos_target": cos, "norm_factor": 1.3})])
    for name in ("S7a", "S7b", "S7bm", "S8t", "S8m", "S8b", "SC1", "SC2", "SC3"):
        S[name] = get_preset_a1(name)

    # --- expansion (review round 3): 17 more schedule TYPES so the design is
    # schedule-rich (34 types x 10 seeds) rather than seed-rich (17 x 20) --
    # doubles the effective N the cross-schedule headline actually needs. --
    def dj(a, **kw): return Event(20, "direction_jump", {"angle_deg": a, "rho_target": 1.0, **kw})
    for a in (15, 35):                                   # intermediate rotation magnitudes
        S[f"S1_{a}"] = Schedule(f"S1_{a}", "A1", [dj(a)])
    for cos in (0.8, 0.6):                                # intermediate swap magnitudes
        S[f"S5_{int(round(100*cos))}"] = Schedule(f"S5_{int(round(100*cos))}", "A1",
            [Event(20, "family_swap", {"cos_target": cos, "norm_factor": 1.3})])
    for fac in (0.99, 0.98):                              # intermediate decays
        S[f"S2_{int(round(1000*fac))}"] = Schedule(f"S2_{int(round(1000*fac))}", "A1",
            [], {"norm_decay": {"from": 20, "to": None, "factor": fac}})
    # mixed direction + norm at the same event
    S["SM1"] = Schedule("SM1", "A1", [dj(25), Event(20, "norm_jump", {"factor": 0.7})])
    S["SM2"] = Schedule("SM2", "A1", [dj(35), Event(20, "norm_jump", {"factor": 1.3})])
    S["SM3"] = Schedule("SM3", "A1", [Event(20, "family_swap", {"cos_target": 0.85, "norm_factor": 0.7})])
    # multi-event calendars
    S["MC1"] = Schedule("MC1", "A1", [Event(12, "direction_jump", {"angle_deg": 15, "rho_target": 1.0}),
                                      Event(28, "direction_jump", {"angle_deg": 15, "rho_target": 1.0})])
    S["MC2"] = Schedule("MC2", "A1", [Event(15, "family_swap", {"cos_target": 0.8, "norm_factor": 1.15}),
                                      Event(30, "norm_jump", {"factor": 0.8})])
    # more behavioral-relevance eigendirections
    for r in (1, 3, 4):
        S[f"S8r{r}"] = Schedule(f"S8r{r}", "A1", [dj(25, eig_rank=r)])
    # criterion-edit combinations
    S["SC4"] = Schedule("SC4", "A1", [Event(20, "criterion_scale", {"feature": 3, "factor": 1.5})])  # water
    S["SC5"] = Schedule("SC5", "A1", [Event(20, "rubric_edit", {"edits": [(2, 0.7), (5, 1.5)]})])     # lava down, bump up
    S["SC6"] = Schedule("SC6", "A1", [Event(20, "tradeoff_rotation",
                                            {"feature_a": 1, "feature_b": 2, "angle_deg": 25})])       # goalprox<->lava
    return S


# S7bm (matched-away robustness cell) is run but held OUT of the predictor hierarchy;
# it is used only for the matched S7 contrast. Everything else is a drift schedule.
DRIFT_SCHEDS = [s for s in e4a_schedules() if s not in ("S0", "S7null", "S7bm")]


def _control(name: str) -> str:
    return "S7null" if name in ("S7a", "S7b") else "S0"


def _one(job):
    out_root, name, seed = job
    sched = e4a_schedules()[name]
    run_a1(name, seed, run_dir=Path(out_root) / name / str(seed),
           estimator_mode="informed", schedule=sched)
    return (name, seed)


def run(out_root, seeds, nproc=1):
    conds = list(e4a_schedules().keys())
    jobs = [(out_root, name, seed) for name in conds for seed in seeds
            if not (Path(out_root) / name / str(seed) / "log.jsonl").exists()]
    print(f"{len(jobs)} jobs", flush=True)
    if nproc > 1:
        import multiprocessing as mp
        with mp.Pool(nproc) as pool:
            for r in pool.imap_unordered(_one, jobs):
                print("done", r, flush=True)
    else:
        for j in jobs:
            print("done", _one(j), flush=True)


def _load(out_root, name, seed):
    rows = [json.loads(l) for l in
            (Path(out_root) / name / str(seed) / "log.jsonl").read_text().splitlines()]
    return {k: np.array([r[k] for r in rows]) for k in
            ("m_k", "gold_wstar", "regret_wt", "e_shift", "e_epi")}


def _first_event(name):
    # first round the schedule induces any drift (harm-window start); handles
    # multi-event calendars (MC*) and continuous S7 correctly.
    return e4a_schedules()[name].first_event_round()


def _boot_ci(vals, rng, n=2000, lo=5, hi=95):
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    s = vals[idx].mean(axis=1)
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


def analyze(out_root, seeds, n_boot=4000):
    rows = []
    for name in DRIFT_SCHEDS:
        ctrl = _control(name)
        fe = _first_event(name)
        for seed in seeds:
            run_ = _load(out_root, name, seed)
            c = _load(out_root, ctrl, seed)
            G = float(np.mean(c["gold_wstar"][fe - 1:] - run_["gold_wstar"][fe - 1:]))
            R = float(np.mean(run_["regret_wt"][fe - 1:]))
            nd, nl, na, da = _features(run_)
            rows.append(dict(name=name, seed=seed, G=G, R=R, net_disp=nd, net_lognorm=nl,
                             net_angle=na, dalign=da,
                             e_shift_int=float(np.mean(run_["e_shift"][fe - 1:])),
                             e_epi_int=float(np.mean(run_["e_epi"][fe - 1:]))))
    rng = np.random.default_rng(0)

    # ---- per-schedule harm table ----
    print(f"{'schedule':9s} {'G_auc':>8s} {'sd':>6s} {'R_auc':>7s} {'net_disp':>9s} {'dalign':>8s}")
    for name in DRIFT_SCHEDS:
        rs = [r for r in rows if r["name"] == name]
        print(f"{name:9s} {np.mean([r['G'] for r in rs]):>8.3f} {np.std([r['G'] for r in rs]):>6.2f} "
              f"{np.mean([r['R'] for r in rs]):>7.3f} {np.mean([r['net_disp'] for r in rs]):>9.3f} "
              f"{np.mean([r['dalign'] for r in rs]):>8.2f}")

    # ---- predictor hierarchy on Y = G_auc ----
    y = np.array([r["G"] for r in rows])
    tiers = {"H0": ["net_disp"],
             "H1": ["net_disp", "net_lognorm", "net_angle"],
             "H2": ["net_disp", "net_lognorm", "net_angle", "dalign"],
             "H3": ["e_shift_int", "e_epi_int"]}
    X = {k: np.array([[r[c] for c in cols] for r in rows]) for k, cols in tiers.items()}
    r2 = {k: _adj_r2(X[k], y) for k in tiers}
    print("\nadjusted R^2:", {k: round(v, 3) for k, v in r2.items()})

    # bootstrap CI of the H2 - H0 increment (seed-level)
    seeds_arr = np.array(sorted(seeds))
    incs = []
    for _ in range(n_boot):
        pick = rng.choice(seeds_arr, size=len(seeds_arr), replace=True)
        idx = [i for s_ in pick for i, r in enumerate(rows) if r["seed"] == s_]
        yy = y[idx]
        if np.std(yy) < 1e-9:
            continue
        incs.append(_adj_r2(X["H2"][idx], yy) - _adj_r2(X["H0"][idx], yy))
    lo, hi = np.percentile(incs, [5, 95])
    print(f"H2-H0 incremental adj-R^2 90% CI [{lo:+.3f}, {hi:+.3f}] -> "
          f"{'PASS (H2>H0)' if lo > 0 else 'FAIL'}")

    # ---- named contrasts ----
    def cell(name, key="G"):
        return np.array([r[key] for r in rows if r["name"] == name])
    Ga, Gb = cell("S7a"), cell("S7b")
    nda, ndb = cell("S7a", "net_disp"), cell("S7b", "net_disp")
    ci_a, ci_b = _boot_ci(Ga, rng), _boot_ci(Gb, rng)
    ci_mag = _boot_ci(nda - ndb, rng)
    print(f"\nS7 contrast: G(S7a)={Ga.mean():+.3f} CI{tuple(round(x,3) for x in ci_a)}  "
          f"G(S7b)={Gb.mean():+.3f} CI{tuple(round(x,3) for x in ci_b)}")
    print(f"  S7a<0<S7b: {'PASS' if ci_a[1] < 0 < ci_b[0] else 'FAIL'} | "
          f"equal magnitude |Δnet_disp| CI {tuple(round(x,3) for x in ci_mag)} "
          f"{'(includes 0)' if ci_mag[0] < 0 < ci_mag[1] else '(excludes 0)'}")
    Gt, Gm, Gb8 = cell("S8t"), cell("S8m"), cell("S8b")
    print(f"S8 ordering: G(S8t)={Gt.mean():.3f} G(S8m)={Gm.mean():.3f} G(S8b)={Gb8.mean():.3f}")
    ci_tb = _boot_ci(Gb8 - Gt, rng)
    print(f"  S8b-S8t CI {tuple(round(x,3) for x in ci_tb)} -> "
          f"{'INVERTED (S8b>S8t)' if ci_tb[0] > 0 else 'as-registered (S8t>S8b)' if ci_tb[1] < 0 else 'no clear ordering'}")
    return rows


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
