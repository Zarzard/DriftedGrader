"""Stage P1 analysis from runs/p1_rows.parquet: C_t trajectories, the quadrant
plot, transient minima, rank validity, and the HG1/HG2/HG3 verdicts.

Run:  python -m analysis.p1_analyze
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path="runs/p1_rows.parquet"):
    return pd.read_parquet(path)


def cell_traj(df, k, lam, breadth, col):
    g = df[(df.k == k) & (df.lam == lam) & (df.breadth == breadth)]
    return g.groupby("t")[col].mean()


def main(path="runs/p1_rows.parquet", figdir="figs"):
    import os
    df = load(path)
    ks = sorted(df.k.unique()); lams = sorted(df.lam.unique())
    breadths = sorted(df.breadth.unique())
    n_seeds = df.seed.nunique()
    T = int(df.t.max()) + 1
    print(f"Stage P1: {n_seeds} seeds, k={ks}, lam={lams}, breadth={breadths}\n")

    # ---- coverage summary: final + transient minimum, per cell ----
    print(f"{'k':>2} {'lam':>4} {'breadth':>9} | {'cov_final':>9} {'cov_MIN':>7} {'W_final':>7} "
          f"{'K_final':>7} {'R_final':>7} {'rankcorr':>8}")
    quad = {}
    for k in ks:
        for lam in lams:
            for br in breadths:
                ct = cell_traj(df, k, lam, br, "hit_0.05")
                if ct.empty:
                    continue
                wt = cell_traj(df, k, lam, br, "W_t")
                kt = cell_traj(df, k, lam, br, "K_t")
                rt = cell_traj(df, k, lam, br, "R_t")
                rc = cell_traj(df, k, lam, br, "rank_corr")
                cov_f = ct.iloc[-5:].mean(); cov_min = ct.iloc[5:].min()
                quad[(k, lam, br)] = (ct.values, wt.values)
                print(f"{k:>2} {lam:>4} {br:>9} | {cov_f:>9.3f} {cov_min:>7.3f} "
                      f"{wt.iloc[-5:].mean():>7.3f} {kt.iloc[-5:].mean():>7.4f} "
                      f"{rt.iloc[-5:].mean():>7.3f} {rc.iloc[-5:].mean():>8.3f}")

    # ---- HG3: drift coverage drop, misspec vs well-spec (on-policy, alpha .05) ----
    print("\nHG3 (on-policy): drift-induced coverage drop, well-spec (k=d) vs misspec (k<d)")
    lam_hi = max(lams)
    for k in ks:
        c0 = cell_traj(df, k, 0.0, "onpolicy", "hit_0.05")
        c1 = cell_traj(df, k, lam_hi, "onpolicy", "hit_0.05")
        if c0.empty or c1.empty:
            continue
        drop_final = c0.iloc[-5:].mean() - c1.iloc[-5:].mean()
        drop_min = c0.iloc[5:].min() - c1.iloc[5:].min()
        print(f"  k={k}: cov_final {c0.iloc[-5:].mean():.3f}->{c1.iloc[-5:].mean():.3f} "
              f"(drop {drop_final:+.3f}); cov_MIN {c0.iloc[5:].min():.3f}->{c1.iloc[5:].min():.3f} "
              f"(drop {drop_min:+.3f})")

    # ---- Figure: C_t trajectories (on-policy) ----
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 3.6), squeeze=False)
    for ax, k in zip(axes[0], ks):
        for lam in lams:
            ct = cell_traj(df, k, lam, "onpolicy", "hit_0.05")
            if not ct.empty:
                ax.plot(ct.index, ct.values, label=f"lam={lam}")
        ax.axhline(0.95, color="k", ls=":", lw=0.8)
        ax.set_title(f"k={k} ({'well-spec' if k >= 6 else 'misspec'}), on-policy")
        ax.set_xlabel("round t"); ax.set_ylabel("coverage C_t (alpha=0.05)")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    fig.suptitle(f"Fig 1: VaR coverage vs round ({n_seeds} seeds)")
    fig.tight_layout(); os.makedirs(figdir, exist_ok=True)
    fig.savefig(f"{figdir}/fig1_coverage.png", dpi=130); plt.close(fig)

    # ---- Quadrant plot: (C_t, W_t) trajectory arrows, on-policy ----
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
    for ax, k in zip(axes[0], ks):
        for lam in lams:
            key = (k, lam, "onpolicy")
            if key not in quad:
                continue
            C, W = quad[key]
            ax.plot(C, W, "-", alpha=0.6, label=f"lam={lam}")
            ax.plot(C[-1], W[-1], "o")
        ax.axvline(0.95, color="k", ls=":", lw=0.8)
        ax.set_xlabel("coverage C_t"); ax.set_ylabel("width W_t")
        ax.set_title(f"k={k}: quadrant (want: stay right)")
        ax.legend(fontsize=7)
    fig.suptitle("Fig 2: dangerous quadrant = low C_t + low W_t (bottom-left)")
    fig.tight_layout(); fig.savefig(f"{figdir}/fig2_quadrant.png", dpi=130); plt.close(fig)
    print(f"\nwrote {figdir}/fig1_coverage.png, {figdir}/fig2_quadrant.png")


if __name__ == "__main__":
    main()
