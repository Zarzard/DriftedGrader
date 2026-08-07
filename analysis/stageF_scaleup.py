"""Stage F scale-up analysis: aggregate rank-correlation AND argmax-VaR
selection-regret across the multi-task x multi-seed x {NONE, SAT} drift runs.

Consumes the exp tree produced by NeuralDriftedGrader/stageF_scaleup:
    <root>/exp/<task>/<arm>/seed<seed>/{ranking_export.pkl, drift_log.jsonl}
where arm in {NONE, SAT_lam0.5, SAT_lam1.0}.

Per run, at a sweep of feedback-round cut-offs t (using rounds[len//3 .. -1]):
  * eval pool  = achieved feature profiles {m_0 .. m_t}
  * v_true(t)  = evalset @ w_t         (true value under the CURRENT grader)
  * lb(t)      = reward-posterior VaR lower bound over the pool, per condition
  * rank_corr  = Spearman(lb, v_true)                    (does the bound RANK right?)
  * sel_regret = normalized regret of the argmax-lb pick  (does it PICK right?)
Conditions (all reuse analysis.stageF_ranking): full / kproj / adv (linear), and an
optional nonlinear MLP-ensemble axis `mlp` (reported only if it passes the NONE>0 gate,
per STAGE_F_HANDOVER caveat #7).

Reuses the validated tabular machinery via analysis.stageF_ranking.

Run:  python -m analysis.stageF_scaleup \
        --root /uufs/.../dbrown-group1/zifan/stageF_scaleup --k 3 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from drifted_grader.metrics import rank_correlation
from drifted_grader.reward_models import make_projection, _apply_P
from drifted_grader.posterior import RewardPosterior
from analysis.stageF_ranking import (
    _load, _pool, _adv_projection, _fit_linear, BETAT, ALPHA, PRIOR, NS,
)

LINEAR_CONDS = ("full", "kproj", "adv")
# nonlinear axes: mlp = ensemble mean-1sigma (legacy); mlp_q05 = ensemble
# mean-1.645sigma (Gaussian alpha=0.05, matched tail); mlp_bll = Bayesian last
# layer -- SAME RewardPosterior VaR machinery (betaT, prior, NS, alpha) as the
# linear conditions, on a single MLP's learned hidden features (fully matched).
MLP_CONDS = ("mlp", "mlp_q05", "mlp_bll", "mlp_bll_kproj", "mlp_bll_adv")
Z_ALPHA = 1.6448536269514722   # Phi^{-1}(1-0.05); mean - Z*sigma ~ 5th pctile (Gaussian)


def _sel_regret(lb, v_true):
    """Normalized selection regret of the argmax-VaR pick in [0,1]."""
    lb = np.asarray(lb, float); v = np.asarray(v_true, float)
    if len(v) < 2:
        return np.nan
    rng = float(v.max() - v.min())
    if rng < 1e-9:
        return 0.0
    pick = int(np.argmax(lb))
    return float((v.max() - v[pick]) / rng)


def _drift_sanity(path):
    """Last-round + 2nd-half-mean drift diagnostics from drift_log.jsonl."""
    p = os.path.join(path, "drift_log.jsonl")
    if not os.path.exists(p):
        return {}
    recs = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    if not recs:
        return {}
    # A re-run (cancel/preempt + restart) APPENDS to drift_log.jsonl while step
    # resets to 0; keep only the final run's segment so sanity stats aren't
    # contaminated by an aborted partial run. (ranking_export.pkl is already
    # rewritten fresh each round, so the core metrics are unaffected.)
    start = 0
    for i in range(1, len(recs)):
        if int(recs[i].get("step", 0)) < int(recs[i - 1].get("step", 0)):
            start = i
    recs = recs[start:]
    half = recs[len(recs) // 2:]

    def m(key):
        vals = [r[key] for r in half if key in r]
        return float(np.mean(vals)) if vals else np.nan
    last = recs[-1]
    return dict(
        n_rounds=len(recs),
        w_dist_last=float(last.get("w_dist_from_w0", np.nan)),
        w_dist_mean2h=m("w_dist_from_w0"),
        J_init_mean2h=m("J_init"),
        J_current_mean2h=m("J_current"),
        w_argmax_last=last.get("w_argmax", None),
    )


def analyze_run(path, k=3, stages=12, with_mlp=False):
    """Return per-condition lists of (rank_corr, sel_regret) across stage cut-offs."""
    try:
        rounds = _load(path)
    except Exception:
        return None
    if len(rounds) < 4:
        return None
    feats = np.stack([r["m"] for r in rounds])
    alls1 = np.concatenate([r["sphi1"] for r in rounds])
    alls2 = np.concatenate([r["sphi2"] for r in rounds])
    scale = 1.0 / (np.sqrt(((alls1 - alls2) ** 2).sum(1).mean()) + 1e-8)
    u = rounds[-1]["w"] - rounds[0]["w"]
    rng = np.random.default_rng(0)
    P_adv = (_adv_projection(feats.shape[1], k, u, rng) if np.linalg.norm(u) > 1e-6
             else make_projection(feats.shape[1], k, rng))
    P_kproj = make_projection(feats.shape[1], k, np.random.default_rng(777))  # run-level generic k<d
    idx = np.unique(np.linspace(len(rounds) // 3, len(rounds) - 1, stages).astype(int))

    conds = list(LINEAR_CONDS) + (list(MLP_CONDS) if with_mlp else [])
    out = {c: {"rank": [], "reg": []} for c in conds}
    for t in idx:
        s1, s2, y = _pool(rounds, t)
        if s1 is None or len(y) < 20:
            continue
        dphi = s1 - s2
        w_t = rounds[t]["w"]; evalset = feats[: t + 1]
        v_true = (evalset @ w_t).tolist()
        rng2 = np.random.default_rng(int(t) + 1)
        for kind in LINEAR_CONDS:
            lb = _fit_linear(dphi, y, evalset, kind, k, scale, rng2, Pmat=P_adv)
            out[kind]["rank"].append(rank_correlation(lb.tolist(), v_true))
            out[kind]["reg"].append(_sel_regret(lb, v_true))
        if with_mlp:
            preds = _ensemble_preds(s1, s2, y, evalset, scale, rng2)   # [K, n_eval]
            mean = preds.mean(0); sd = preds.std(0)
            for name, lbv in (("mlp", mean - sd), ("mlp_q05", mean - Z_ALPHA * sd)):
                out[name]["rank"].append(rank_correlation(lbv.tolist(), v_true))
                out[name]["reg"].append(_sel_regret(lbv, v_true))
            # nonlinear 2x3 axis: same BLL machinery, input projected full / generic-k / drift-blind-k
            for name, Pm in (("mlp_bll", None), ("mlp_bll_kproj", P_kproj), ("mlp_bll_adv", P_adv)):
                lb_bll = _fit_mlp_bll(s1, s2, y, evalset, scale, rng2, P=Pm)
                out[name]["rank"].append(rank_correlation(lb_bll.tolist(), v_true))
                out[name]["reg"].append(_sel_regret(lb_bll, v_true))
    return out


def _ensemble_preds(s1, s2, y, feats, scale, rng, K=5, hidden=32):
    """EnsembleBT of MLP heads (base='mlp', full features P=None) fit on the
    per-segment BT comparison set; returns per-head predictions on the eval pool
    [K, n_eval]. Bounds (mean-sigma, mean-Z*sigma) are derived by the caller."""
    from drifted_grader.reward_models import EnsembleBT
    d = s1.shape[1]
    Phi = np.vstack([s1 * scale, s2 * scale])          # [2N, d]
    n = len(y)
    pairs = np.stack([np.arange(n), np.arange(n) + n], axis=1)   # (seg1_i, seg2_i)
    ens = EnsembleBT(d, P=None, K=K, reg=1e-3, base="mlp", hidden=hidden, steps=150)
    ens.fit_buffer(Phi, pairs, np.asarray(y, float), rng)        # y=1 => seg1 preferred
    return ens.predict_all(feats * scale)              # [K, n_eval]


def _fit_mlp_bll(s1, s2, y, feats, scale, rng, hidden=6, P=None):
    """Bayesian last layer (matched nonlinear VaR), recalibrated to be a FAIR match to
    the linear conditions: (a) hidden dim = d (=6) so the posterior lives in the same
    dimension as the linear `full` condition; (b) STANDARDIZE the learned hidden
    features phi_h(x)=tanh(W1 [P]x + b1) to zero-mean/unit-variance (using training-segment
    stats) so prior_scale/betaT are in-scale. Then run the SAME RewardPosterior
    posterior-VaR machinery (betaT, prior_scale, NS, alpha).

    P projects the INPUT before the MLP: P=None -> full features (nonlinear analog of
    `full`); P=random k<d -> `mlp_bll_kproj`; P=P_adv (rows in u^perp) -> `mlp_bll_adv`,
    a nonlinear reward that is PROVABLY BLIND to the drift axis u (the coordinate is
    annihilated at the input, so no nonlinearity downstream can recover it)."""
    from drifted_grader.reward_models import MLPBT
    d = s1.shape[1]
    Phi = np.vstack([s1 * scale, s2 * scale]); n = len(y)
    pairs = np.stack([np.arange(n), np.arange(n) + n], axis=1)
    mlp = MLPBT(d, P=P, hidden=hidden, reg=1e-3, steps=150, seed=0)
    mlp.fit_buffer(Phi, pairs, np.asarray(y, float), rng)
    raw = lambda X: np.tanh(_apply_P(np.asarray(X) * scale, P) @ mlp.W1.T + mlp.b1)  # [., hidden]
    # (b) standardize hidden features on the training segments (mean cancels in the
    # difference; the constant offset only shifts all policy values -> rank-invariant).
    Htr = np.vstack([raw(s1), raw(s2)])
    mu = Htr.mean(0); sd = Htr.std(0) + 1e-8
    hfun = lambda X: (raw(X) - mu) / sd
    dphi_h = hfun(s1) - hfun(s2)
    F_h = hfun(feats)
    post = RewardPosterior(hidden)
    post.fit(dphi_h, np.asarray(y, float), BETAT, prior_scale=PRIOR, n_samples=NS, rng=rng)
    return np.array([post.lower_bound(f, ALPHA) for f in F_h])


def _parse_path(path):
    # <root>/exp/<task>/<arm>/seed<seed>
    parts = path.rstrip("/").split(os.sep)
    seed = int(parts[-1].replace("seed", "")) if parts[-1].startswith("seed") else -1
    arm = parts[-2]; task = parts[-3]
    return task, arm, seed


def _worker(args):
    path, k, with_mlp = args
    task, arm, seed = _parse_path(path)
    res = analyze_run(path, k=k, with_mlp=with_mlp)
    san = _drift_sanity(path)
    return dict(path=path, task=task, arm=arm, seed=seed, res=res, san=san)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="stageF_scaleup root (expects <root>/exp/<task>/<arm>/seed*)")
    ap.add_argument("--out", default=None, help="output dir (default <root>/results)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--with_mlp", action="store_true",
                    help="also run the nonlinear MLP-ensemble axis (caveat #7)")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.root, "results")
    os.makedirs(out_dir, exist_ok=True)

    run_dirs = sorted(glob.glob(os.path.join(args.root, "exp", "*", "*", "seed*")))
    run_dirs = [p for p in run_dirs if os.path.exists(os.path.join(p, "ranking_export.pkl"))]
    print(f"found {len(run_dirs)} runs with ranking_export.pkl")
    if not run_dirs:
        return

    tasks_args = [(p, args.k, args.with_mlp) for p in run_dirs]
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_worker, tasks_args):
                results.append(r)
    else:
        results = [_worker(a) for a in tasks_args]

    _write_outputs(results, out_dir, args.with_mlp)


def _write_outputs(results, out_dir, with_mlp):
    import csv
    conds = list(LINEAR_CONDS) + (list(MLP_CONDS) if with_mlp else [])
    # ---- per-run CSV (run-mean of each metric over stages) ----
    per_run_path = os.path.join(out_dir, "per_run.csv")
    rows = []
    for r in results:
        if r["res"] is None:
            continue
        base = dict(task=r["task"], arm=r["arm"], seed=r["seed"], **r["san"])
        for c in conds:
            rk = r["res"][c]["rank"]; rg = r["res"][c]["reg"]
            if not rk:
                continue
            row = dict(base)
            row.update(condition=c,
                       rank_corr=float(np.nanmean(rk)),
                       sel_regret=float(np.nanmean(rg)),
                       n_stages=len(rk))
            rows.append(row)
    if rows:
        keys = ["task", "arm", "seed", "condition", "rank_corr", "sel_regret",
                "n_stages", "n_rounds", "w_dist_last", "w_dist_mean2h",
                "J_init_mean2h", "J_current_mean2h", "w_argmax_last"]
        with open(per_run_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"wrote {per_run_path} ({len(rows)} rows)")

    # ---- aggregate: mean +/- sem over seeds, per (task, arm, condition) ----
    agg = defaultdict(lambda: defaultdict(list))  # (task,arm,cond) -> metric -> [per-seed means]
    for row in rows:
        key = (row["task"], row["arm"], row["condition"])
        agg[key]["rank"].append(row["rank_corr"])
        agg[key]["sel"].append(row["sel_regret"])

    def ms(v):
        v = np.asarray(v, float); v = v[~np.isnan(v)]
        if len(v) == 0:
            return (np.nan, np.nan, 0)
        sem = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        return (float(np.mean(v)), sem, len(v))

    tasks = sorted({r["task"] for r in results})
    arms = ["NONE", "SAT_lam0.5", "SAT_lam1.0"]

    lines = ["# Stage F scale-up results\n",
             "Selection regret (argmax-VaR pick, normalized [0,1], lower=better) and "
             "rank correlation (Spearman LB vs true value, higher=better), "
             "mean +/- sem over seeds.\n"]
    # markdown tables per metric
    for metric, label, key in [("sel", "Selection regret (lower better)", "sel"),
                               ("rank", "Rank correlation (higher better)", "rank")]:
        lines.append(f"\n## {label}\n")
        header = "| task | condition | " + " | ".join(arms) + " |"
        sep = "|" + "---|" * (2 + len(arms))
        lines.append(header); lines.append(sep)
        for task in tasks:
            for c in conds:
                cells = []
                for arm in arms:
                    m, s, n = ms(agg[(task, arm, c)][key])
                    cells.append("n/a" if n == 0 else f"{m:+.3f}±{s:.3f} (n={n})")
                lines.append(f"| {task} | {c} | " + " | ".join(cells) + " |")
        # pooled across tasks
        lines.append(f"| **POOLED** | | | | |")
        for c in conds:
            cells = []
            for arm in arms:
                pooled = [row[f"{'sel_regret' if key=='sel' else 'rank_corr'}"]
                          for row in rows if row["arm"] == arm and row["condition"] == c]
                m, s, n = ms(pooled)
                cells.append("n/a" if n == 0 else f"{m:+.3f}±{s:.3f} (n={n})")
            lines.append(f"| POOLED | {c} | " + " | ".join(cells) + " |")

    # ---- drift sanity summary ----
    lines.append("\n## Drift sanity (2nd-half means over seeds)\n")
    lines.append("| task | arm | w_dist | J_init | J_current | n |")
    lines.append("|---|---|---|---|---|---|")
    san_agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r["san"]:
            continue
        k = (r["task"], r["arm"])
        for f in ("w_dist_mean2h", "J_init_mean2h", "J_current_mean2h"):
            if f in r["san"] and not (isinstance(r["san"][f], float) and np.isnan(r["san"][f])):
                san_agg[k][f].append(r["san"][f])
    for task in tasks:
        for arm in arms:
            wd = san_agg[(task, arm)]["w_dist_mean2h"]
            ji = san_agg[(task, arm)]["J_init_mean2h"]
            jc = san_agg[(task, arm)]["J_current_mean2h"]
            if not wd:
                continue
            lines.append(f"| {task} | {arm} | {np.mean(wd):.3f} | {np.mean(ji):.3f} "
                         f"| {np.mean(jc):.3f} | {len(wd)} |")

    md_path = os.path.join(out_dir, "results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {md_path}")
    _make_figures(rows, out_dir, arms, conds)


def _make_figures(rows, out_dir, arms, conds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("skip figures (matplotlib unavailable):", e)
        return
    tasks = sorted({r["task"] for r in rows})
    for metric, ylab, key in [("sel_regret", "selection regret (lower=better)", "sel_regret"),
                              ("rank_corr", "rank corr (higher=better)", "rank_corr")]:
        fig, axes = plt.subplots(1, len(tasks) + 1, figsize=(4 * (len(tasks) + 1), 4),
                                 sharey=True)
        panels = list(tasks) + ["POOLED"]
        for ax, panel in zip(np.atleast_1d(axes), panels):
            x = np.arange(len(conds)); width = 0.26
            for i, arm in enumerate(arms):
                means, sems = [], []
                for c in conds:
                    vals = [r[key] for r in rows
                            if r["condition"] == c and r["arm"] == arm
                            and (panel == "POOLED" or r["task"] == panel)]
                    vals = [v for v in vals if not np.isnan(v)]
                    means.append(np.mean(vals) if vals else np.nan)
                    sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                                if len(vals) > 1 else 0.0)
                ax.bar(x + (i - 1) * width, means, width, yerr=sems, capsize=3, label=arm)
            ax.set_xticks(x); ax.set_xticklabels(conds, rotation=0)
            ax.set_title(panel.replace("metaworld_", ""), fontsize=9)
            ax.axhline(0, color="k", lw=0.5)
        np.atleast_1d(axes)[0].set_ylabel(ylab)
        np.atleast_1d(axes)[-1].legend(fontsize=8)
        fig.suptitle(f"Stage F scale-up: {metric} by misspec condition x drift arm")
        fig.tight_layout()
        fp = os.path.join(out_dir, f"fig_{metric}.png")
        fig.savefig(fp, dpi=130); plt.close(fig)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
