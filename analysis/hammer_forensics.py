"""Forensics: WHY does the correctly-specified (`full`) linear posterior degrade on
some tasks under drift (hammer rank 1.00 -> 0.60 -> 0.35) when it has no span gap?

NOTE ON THE TEACHER: these runs use teacher_beta=-1 (deterministic argmax labels,
zero noise/ties), while the posterior is fit with a betaT=10 BT likelihood. So
"noisy labels" is NOT a candidate mechanism; the label-side candidate is
LIKELIHOOD MISMATCH ON NEAR-MARGIN PAIRS (deterministic labels on tiny margins
carry almost no usable constraint under the BT likelihood).

Candidate mechanisms, each measured per run from ranking_export.pkl (no refits):
  H-A value-spread collapse : drifted w_t shrinks the true-value spread across the
      eval pool -> rank corr on near-ties is noise.       metric: spread_ratio
      (range-based), spread_ratio_std (robust), tie_frac
  H-B eval-pool degeneracy  : achieved profiles {m_t} barely differ (converged
      policy) -> ranking near-identical policies.          metric: eval_disp,
      eval_effrank (participation ratio)
  H-C labeling-margin collapse: per-round comparison margins under that round's
      grader shrink over training (deterministic teacher, so this starves the
      BT fit of constraint, not adds noise).               metric: sig_ratio
  H-D drift-direction data coverage: pooled dphi variance along u vs typical.
      metric: u_cov_ratio = u^T Cov(dphi) u / median eigenvalue
  H-D' final-grader data coverage: same along w_T.         metric: w_cov_ratio
  H-E eval-direction data coverage: dphi variance along the direction that
      actually separates the eval policies.                metric: e1_cov_ratio

Cross-task test: Spearman (scipy, tie-corrected, finite-masked, permutation p)
between each metric and the run's `full` rank_corr (joined from
results/per_run.csv), pooled / SAT-only / per-task means / within-hammer
across-seed. The winning metric is the mechanism candidate; n=10 tasks is low
power, so treat the per-task column as suggestive and lean on the within-task
and per-run columns.

Run: python -m analysis.hammer_forensics --root <stageF_scaleup root> [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats as sps

from analysis.stageF_ranking import _load, _pool, BETAT


def _spear(x, y, nperm=0, rng=None):
    """Tie-corrected Spearman on finite pairs; optional permutation p-value."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    r, _ = sps.spearmanr(x[ok], y[ok])
    p = np.nan
    if nperm and np.isfinite(r):
        rng = rng or np.random.default_rng(0)
        null = np.array([sps.spearmanr(rng.permutation(x[ok]), y[ok])[0]
                         for _ in range(nperm)])
        p = float((np.abs(null) >= abs(r)).mean())
    return float(r), p, int(ok.sum())


def _effrank(X):
    """Participation ratio of the centered covariance eigenvalues."""
    if len(X) <= 2:
        return 1.0
    ev = np.clip(np.linalg.eigvalsh(np.cov((X - X.mean(0, keepdims=True)).T)), 0, None)
    s = ev.sum()
    return float(s * s / ((ev ** 2).sum() + 1e-18)) if s > 1e-18 else 0.0


def forensics_run(path, stages=12):
    try:
        rounds = _load(path)
    except Exception:
        return None
    if len(rounds) < 4:
        return None
    feats = np.stack([r["m"] for r in rounds])
    w0, wT = rounds[0]["w"], rounds[-1]["w"]
    u = wT - w0
    unorm = float(np.linalg.norm(u))
    idx = np.unique(np.linspace(len(rounds) // 3, len(rounds) - 1, stages).astype(int))

    spread_ratios, spread_ratios_std, tie_fracs = [], [], []
    for t in idx:
        # mirror stageF_scaleup.analyze_run's stage gate exactly (len(y) >= 20)
        s1, s2, y = _pool(rounds, t)
        if s1 is None or len(y) < 20:
            continue
        w_t = rounds[t]["w"]
        evalset = feats[: t + 1]
        v0 = evalset @ w0
        vt = evalset @ w_t
        r0 = float(v0.max() - v0.min()) + 1e-12
        spread_ratios.append(float(vt.max() - vt.min()) / r0)
        spread_ratios_std.append(float(np.std(vt) / (np.std(v0) + 1e-12)))
        dv = np.abs(vt[:, None] - vt[None, :])
        iu = np.triu_indices(len(vt), 1)
        tie_fracs.append(float((dv[iu] < 0.05 * r0).mean()) if len(iu[0]) else np.nan)
    if not spread_ratios:
        return None

    # pooled comparison-data geometry (final stage)
    s1, s2, y = _pool(rounds, len(rounds) - 1)
    if s1 is None:
        return None
    dphi = s1 - s2
    C = np.cov(dphi.T)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-18, None)
    med_ev = float(np.median(ev))
    uhat = u / (unorm + 1e-12)
    u_cov_ratio = float(uhat @ C @ uhat / med_ev) if unorm > 1e-6 else np.nan
    what = wT / (np.linalg.norm(wT) + 1e-12)
    w_cov_ratio = float(what @ C @ what / med_ev)
    Fc = feats - feats.mean(0, keepdims=True)
    _, evec = np.linalg.eigh(np.cov(Fc.T))
    e1 = evec[:, -1]
    e1_cov_ratio = float(e1 @ C @ e1 / med_ev)

    # H-C: per-round labeling MARGIN under that round's own grader (the labels
    # were generated at w_r; deterministic teacher => margin collapse = starved
    # constraint, not noise)
    half = len(rounds) // 2
    def _sig(rounds_slice):
        vals = []
        for r in rounds_slice:
            keep = r["lbl"] >= 0
            if keep.sum() >= 5:
                d = r["sphi1"][keep] - r["sphi2"][keep]
                vals.append(BETAT * np.std(d @ r["w"]))
        return float(np.mean(vals)) if vals else np.nan
    sig_ratio = _sig(rounds[half:]) / (_sig(rounds[:half]) + 1e-12)

    W = np.stack([r["w"] for r in rounds])
    argmax_flips = int((np.diff(np.argmax(W, 1)) != 0).sum())

    return dict(
        spread_ratio=float(np.nanmean(spread_ratios)),
        spread_ratio_std=float(np.nanmean(spread_ratios_std)),
        spread_ratio_last=float(spread_ratios[-1]),
        tie_frac=float(np.nanmean(tie_fracs)),
        eval_disp=float(np.mean(np.linalg.norm(Fc, axis=1)) /
                        (np.mean(np.linalg.norm(feats, axis=1)) + 1e-12)),
        eval_effrank=_effrank(feats),
        sig_ratio=float(sig_ratio),
        u_cov_ratio=u_cov_ratio,
        w_cov_ratio=w_cov_ratio,
        e1_cov_ratio=e1_cov_ratio,
        u_norm=unorm,
        argmax_flips=argmax_flips,
        n_rounds=len(rounds),
    )


def _parse_path(path):
    parts = path.rstrip("/").split(os.sep)
    seed = int(parts[-1].replace("seed", "")) if parts[-1].startswith("seed") else -1
    return parts[-3], parts[-2], seed          # task, arm, seed


def _worker(path):
    task, arm, seed = _parse_path(path)
    return dict(task=task, arm=arm, seed=seed, met=forensics_run(path))


METRICS = ("spread_ratio", "spread_ratio_std", "spread_ratio_last", "tie_frac",
           "eval_disp", "eval_effrank", "sig_ratio", "u_cov_ratio", "w_cov_ratio",
           "e1_cov_ratio", "u_norm", "argmax_flips")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nperm", type=int, default=10000)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results_forensics")
    os.makedirs(out_dir, exist_ok=True)

    run_dirs = sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
    run_dirs = [p for p in run_dirs if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    print(f"found {len(run_dirs)} runs", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = [r for r in ex.map(_worker, run_dirs) if r["met"] is not None]

    # join the existing per_run.csv `full` rank_corr (no refits)
    rank = {}
    with open(os.path.join(args.root, "results", "per_run.csv")) as f:
        for row in csv.DictReader(f):
            if row["condition"] == "full":
                rank[(row["task"], row["arm"], int(row["seed"]))] = float(row["rank_corr"])

    rows = []
    for r in results:
        key = (r["task"], r["arm"], r["seed"])
        if key in rank:
            rows.append(dict(task=r["task"], arm=r["arm"], seed=r["seed"],
                             full_rank=rank[key], **r["met"]))
    print(f"joined {len(rows)} rows with per_run.csv full rank_corr", flush=True)

    with open(os.path.join(out_dir, "forensics_per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    rng = np.random.default_rng(0)
    sat_rows = [r for r in rows if r["arm"] != "NONE"]
    tasks = sorted({r["task"] for r in rows})
    HAMMER = "metaworld_hammer-v2"

    lines = ["# Hammer forensics: what breaks the CORRECTLY-SPECIFIED model under drift?\n",
             f"{len(rows)} runs; teacher_beta=-1 (deterministic labels -- noisy-label "
             "mechanisms are excluded a priori; see H-C reframe in the docstring). "
             "`full_rank` = the well-specified linear posterior's rank correlation "
             "from results/per_run.csv. All correlations: tie-corrected Spearman on "
             "finite pairs; p = two-sided permutation. n=10 tasks is LOW POWER -- "
             "lean on the per-run and within-hammer columns.\n",
             "## Which metric predicts full-model degradation?\n",
             "| metric | all runs (n) | SAT only (n) | per-task means, SAT_lam1.0 "
             "(n=10) [perm p] | hammer across-seed, SAT arms (n) |",
             "|---|---|---|---|---|"]
    for m in METRICS:
        ra, _, na = _spear([r[m] for r in rows], [r["full_rank"] for r in rows])
        rs, _, ns = _spear([r[m] for r in sat_rows], [r["full_rank"] for r in sat_rows])
        tx, ty = [], []
        for task in tasks:
            tr = [r for r in rows if r["task"] == task and r["arm"] == "SAT_lam1.0"]
            if tr:
                tx.append(np.nanmean([r[m] for r in tr]))
                ty.append(np.nanmean([r["full_rank"] for r in tr]))
        rt, pt, nt = _spear(tx, ty, nperm=args.nperm, rng=rng)
        hr = [r for r in rows if r["task"] == HAMMER and r["arm"] != "NONE"]
        rh, _, nh = _spear([r[m] for r in hr], [r["full_rank"] for r in hr])
        lines.append(f"| {m} | {ra:+.2f} ({na}) | {rs:+.2f} ({ns}) "
                     f"| {rt:+.2f} ({nt}) [p={pt:.3f}] | {rh:+.2f} ({nh}) |")

    lines += ["\n## Per-task means at SAT_lam1.0 (sorted by full_rank)\n",
              "| task | full_rank | " + " | ".join(METRICS) + " |",
              "|" + "---|" * (2 + len(METRICS))]
    task_rows = []
    for task in tasks:
        tr = [r for r in rows if r["task"] == task and r["arm"] == "SAT_lam1.0"]
        if tr:
            task_rows.append((float(np.nanmean([r["full_rank"] for r in tr])), task, tr))
    for fr, task, tr in sorted(task_rows):
        cells = " | ".join(f"{np.nanmean([r[m] for r in tr]):.3f}" for m in METRICS)
        lines.append(f"| {task.replace('metaworld_','')} | {fr:+.2f} | {cells} |")

    # metric-metric Spearman matrix (are the "mechanisms" independent?)
    lines.append("\n## Metric-metric Spearman (all runs) -- collinear metrics are "
                 "one mechanism, not several\n")
    lines.append("| | " + " | ".join(METRICS) + " |")
    lines.append("|" + "---|" * (1 + len(METRICS)))
    for m1 in METRICS:
        cells = []
        for m2 in METRICS:
            r12, _, _ = _spear([r[m1] for r in rows], [r[m2] for r in rows])
            cells.append(f"{r12:+.2f}")
        lines.append(f"| {m1} | " + " | ".join(cells) + " |")

    lines.append("\n## hammer per-seed (SAT arms)\n")
    lines.append("| arm | seed | full_rank | spread_ratio | spread_ratio_std "
                 "| tie_frac | sig_ratio | w_cov_ratio |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda z: (z["arm"], z["seed"])):
        if r["task"] == HAMMER and r["arm"] != "NONE":
            lines.append(f"| {r['arm']} | {r['seed']} | {r['full_rank']:+.2f} "
                         f"| {r['spread_ratio']:.3f} | {r['spread_ratio_std']:.3f} "
                         f"| {r['tie_frac']:.3f} | {r['sig_ratio']:.3f} "
                         f"| {r['w_cov_ratio']:.3f} |")

    md = "\n".join(lines) + "\n"
    open(os.path.join(out_dir, "results.md"), "w").write(md)
    json.dump(rows, open(os.path.join(out_dir, "forensics.json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
