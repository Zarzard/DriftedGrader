"""Figures from logs (plan Section 7.1). Figures must be regenerable from parquet
logs alone, no re-running. All figures state the seed count.

Stage A:
  Fig 1  : (lambda, 1/beta) phase diagram, one panel per drift model, VR contours
           (mechanism) with ||w_T - w0|| overlaid.
  Fig 1b : the R_bar regret floor as a function of (L, sigma), shown flat in t.
Tandem:
  Fig T  : active vs passive regret over rounds, linear vs MLP (the tandem effect).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(runs_glob: str) -> pd.DataFrame:
    """Load every parquet under a glob into one DataFrame. X5: refuse to mix
    git_sha unless overridden by the caller (checked here as a warning)."""
    files = sorted(glob.glob(runs_glob, recursive=True))
    if not files:
        raise FileNotFoundError(runs_glob)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "git_sha" in df and df["git_sha"].nunique() > 1:
        print(f"WARNING: mixed git_sha in aggregation: {df['git_sha'].unique()}")
    return df


def _cell_agg(df, value, by=("lam", "beta"), stat="mean"):
    g = df.groupby(list(by))[value]
    return (g.mean() if stat == "mean" else g.median()).reset_index()


def fig1_phase_diagram(df, out_path, value="VR_t", drift_col="drift_model"):
    """Panel per drift model; heatmap of second-half mean VR over (lambda, beta)."""
    drifts = sorted(df[drift_col].unique())
    fig, axes = plt.subplots(1, len(drifts), figsize=(4 * len(drifts), 3.4),
                             squeeze=False)
    half = df["t"].max() // 2
    d2 = df[df["t"] > half]
    lams = sorted(df["lam"].unique()); betas = sorted(df["beta"].unique())
    for ax, drift in zip(axes[0], drifts):
        sub = d2[d2[drift_col] == drift]
        M = np.full((len(lams), len(betas)), np.nan)
        for il, lam in enumerate(lams):
            for ib, b in enumerate(betas):
                v = sub[(sub.lam == lam) & (sub.beta == b)][value]
                if len(v):
                    M[il, ib] = np.nanmean(v)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r",
                       vmin=0.2, vmax=1.1)
        ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"{b:g}" for b in betas], rotation=45)
        ax.set_yticks(range(len(lams))); ax.set_yticklabels([f"{l:g}" for l in lams])
        ax.set_title(f"{drift}  (VR; <1 = blind)")
        ax.set_xlabel("beta (pressure ->)"); ax.set_ylabel("lambda")
        fig.colorbar(im, ax=ax, fraction=0.046)
    n_seeds = df["seed"].nunique()
    fig.suptitle(f"Fig 1: mechanism VR phase diagram (2nd-half mean, {n_seeds} seeds)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def fig1b_regret_floor(df, out_path):
    """R_bar floor vs (L, sigma); a small-and-flat-in-t floor is required."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for sigma in sorted(df["sigma"].unique()):
        for L in sorted(df["L"].unique()):
            sub = df[(df.sigma == sigma) & (df.L == L)]
            if not len(sub):
                continue
            curve = sub.groupby("t")["R_t"].mean()
            ax.plot(curve.index, curve.values, label=f"L={L}, sig={sigma}", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("round t"); ax.set_ylabel("R_t (mean over seeds)")
    ax.set_title("Fig 1b: Stage-A regret floor (should be small & flat in t)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def fig_tandem(active_by_round, passive_by_round, labels, out_path):
    """active vs passive regret over rounds, one line-pair per condition label."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for (a, p, lab) in zip(active_by_round, passive_by_round, labels):
        ax.plot(a, label=f"{lab} active", lw=1.5)
        ax.plot(p, "--", label=f"{lab} passive", lw=1.5)
    ax.set_xlabel("round t"); ax.set_ylabel("regret R_t")
    ax.set_title("Tandem effect: passive vs active regret")
    ax.legend(fontsize=7)
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path
