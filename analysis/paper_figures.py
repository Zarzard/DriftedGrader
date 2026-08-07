"""Paper figures for md_files/PAPER.md — regenerable from stored results only.

All four figures read real stored data (no re-running of experiments):
  Fig 1  core dissociation (tabular Plan-02 VaR simulator, §4/§5 aggregates)
  Fig 2  neural transfer / coverage gradient  (results/per_run.csv, n=80)
  Fig 3  exploitability box-regret            (results_hack/per_run.csv)
  Fig 4  the coverage law                     (llm_judge_drift/coverage_law/law.json)

Run (bpref env has pandas+matplotlib):
  python -m analysis.paper_figures
Outputs -> md_files/figures/*.png
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/stageF_scaleup"
LAW = "/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/llm_judge_drift/coverage_law/law.json"
SEM = "/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/llm_judge_drift/semantic/sem.json"
OUT = "/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader/md_files/figures"
os.makedirs(OUT, exist_ok=True)
DRIFTS = ["NONE", "SAT_lam0.5", "SAT_lam1.0"]
DLAB = ["none", "moderate", "strong"]
# colour-blind-safe (Wong)
C = {"full": "#0072B2", "gblind": "#E69F00", "dblind": "#D55E00", "neural": "#009E73",
     "cert": "#0072B2", "rank": "#D55E00", "grey": "#888888"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 150})


# ------------------------------------------------------------------ Fig 1
def fig1_dissociation():
    """Tabular Plan-02 VaR simulator — values are the §4 table aggregates (8-40 seeds).
    Blind model measured at none/moderate/strong; correct model at none/strong."""
    x = [0, 1, 2]
    # blind (k<d): none, moderate, strong
    blind_cert = [0.95, 1.00, 1.00]
    blind_rank = [0.78, 0.08, -0.29]
    blind_width = [0.083, 0.040, 0.034]
    # full (correct): none, strong  (no moderate cell in the table)
    full_cert_x, full_cert = [0, 2], [0.93, 1.00]
    full_rank_x, full_rank = [0, 2], [1.00, 0.98]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    # Panel A: single-policy certification
    axA.axhline(0.95, ls=":", c=C["grey"], lw=1, label="0.95 target")
    axA.plot(x, blind_cert, "-o", c=C["dblind"], lw=2.2, label="blind reward model")
    axA.plot(full_cert_x, full_cert, "--s", c=C["full"], lw=1.8, label="correct reward model")
    axA.set_ylim(0.85, 1.02)
    axA.set_title("Single-policy certification\n(is THIS policy safe?)", fontsize=10.5)
    axA.set_ylabel("coverage  Pr[ true value $\\geq$ bound ]")
    axA.set_xticks(x); axA.set_xticklabels(DLAB)
    axA.set_xlabel("grader drift"); axA.legend(fontsize=8, loc="lower left")
    axA.annotate("stays honest;\ntightens around\na covered value", (2, 1.00),
                 (1.18, 0.905), fontsize=8, color=C["dblind"],
                 arrowprops=dict(arrowstyle="->", color=C["dblind"], lw=1))

    # Panel B: cross-policy ranking
    axB.axhspan(-0.4, 0, color=C["dblind"], alpha=0.06)
    axB.axhline(0, c="k", lw=0.8)
    axB.plot(x, blind_rank, "-o", c=C["dblind"], lw=2.2, label="blind reward model")
    axB.plot(full_rank_x, full_rank, "--s", c=C["full"], lw=1.8, label="correct reward model")
    for xi, r, w, dy in zip(x, blind_rank, blind_width, [-0.15, -0.15, 0.11]):
        axB.annotate(f"width {w:.3f}", (xi, r), (xi, r + dy), ha="center",
                     fontsize=7.5, color=C["grey"])
    axB.set_ylim(-0.45, 1.15)
    axB.set_title("Cross-policy ranking\n(WHICH policy is safest?)", fontsize=10.5)
    axB.set_ylabel("rank correlation (bound vs true value)")
    axB.set_xticks(x); axB.set_xticklabels(DLAB)
    axB.set_xlabel("grader drift"); axB.legend(fontsize=8, loc="center right")
    axB.text(1.0, -0.30, "ranking inverts while the bound tightens\n"
             "(narrower = looks more confident)",
             ha="center", fontsize=8, color=C["dblind"])

    fig.suptitle("A blind reward model's safety bound stays honest for one policy "
                 "but silently inverts when ranking many  (tabular simulator)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_dissociation.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)


# ------------------------------------------------------------------ Fig 2
def fig2_neural_transfer():
    d = pd.read_csv(f"{RES}/results/per_run.csv")
    conds = [("full", "full (correct)", C["full"]),
             ("kproj", "generic-blind", C["gblind"]),
             ("adv", "drift-blind", C["dblind"]),
             ("mlp_bll", "neural (correct)", C["neural"])]
    x = np.arange(3); w = 0.2
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    def bars(ax, col, higher_better):
        for i, (c, lab, col_) in enumerate(conds):
            m = [d[(d.condition == c) & (d.arm == a)][col].mean() for a in DRIFTS]
            e = [d[(d.condition == c) & (d.arm == a)][col].sem() for a in DRIFTS]
            ax.bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=2.5, color=col_, label=lab,
                   error_kw=dict(lw=0.9))
        ax.set_xticks(x); ax.set_xticklabels(DLAB); ax.set_xlabel("grader drift")

    bars(axA, "rank_corr", True)
    axA.axhline(0, c="k", lw=0.6)
    axA.set_ylabel("rank correlation (higher = better)")
    axA.set_title("Ranking reliability")
    axA.annotate("drift-blind\ncollapses", (1 + 0.5 * w, 0.18), (1.35, 0.62),
                 fontsize=8, color=C["dblind"],
                 arrowprops=dict(arrowstyle="->", color=C["dblind"], lw=1))
    bars(axB, "sel_regret", False)
    axB.set_ylabel("selection regret (lower = better)")
    axB.set_title("Selection regret")
    axA.legend(fontsize=8, ncol=2, loc="upper right")

    fig.suptitle("The failure transfers to neural scale — 240 PEBBLE runs, 10 Metaworld "
                 "tasks (pooled n=80/cell, error bars = SEM)", fontsize=11, y=1.01)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_neural_transfer.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)


# ------------------------------------------------------------------ Fig 3
def fig3_exploitability():
    h = pd.read_csv(f"{RES}/results_hack/per_run.csv")
    fams = [("linear head", [("full", "correct"), ("kproj", "generic-blind"),
                             ("adv", "drift-blind")]),
            ("neural head", [("mlp_full", "correct"), ("mlp_kproj", "generic-blind"),
                             ("mlp_adv", "drift-blind")])]
    cols = {"correct": C["full"], "generic-blind": C["gblind"], "drift-blind": C["dblind"]}
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xt, xl = [], []
    pos = 0
    for fname, members in fams:
        for cond, lab in members:
            v = h[h.condition == cond]["box_regret"]
            ax.bar(pos, v.mean(), 0.8, yerr=v.sem(), capsize=3, color=cols[lab],
                   label=lab if pos < 3 else None)
            xt.append(pos); xl.append(lab); pos += 1
        pos += 0.8
    ax.axhline(0.02, ls=":", c=C["grey"], lw=1)
    ax.text(0.1, 0.035, "correct-model floor $\\approx$0.02", fontsize=8, color=C["grey"])
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("box regret  (true value lost by the bound's\nsingle favourite behaviour)")
    ax.text(1.0, -0.20, "linear head", ha="center", transform=ax.get_xaxis_transform(),
            fontsize=9.5, weight="bold")
    ax.text(4.8, -0.20, "neural head", ha="center", transform=ax.get_xaxis_transform(),
            fontsize=9.5, weight="bold")
    ax.legend(fontsize=8, title="representation", loc="upper left")
    ax.set_title("Every blind bound picks a genuinely-bad behaviour — in both model classes\n"
                 "(so it is coverage, not linear-vs-neural)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_exploitability.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)


# ------------------------------------------------------------------ Fig 4
def fig4_coverage_law():
    J = json.load(open(LAW))
    rows = J["rows"]
    cov = np.array([r["covR2"] for r in rows])
    jr = np.array([r["judge_rank"] for r in rows])
    orc = np.array([r["oracle_rank"] for r in rows])
    spear = J["spearman_covR2_vs_judgerank"]
    null_mean = J["mean_null_r2"]
    # first 6 rows are the pure interpretable attributes, in this order:
    attr = ["length", "sentiment", "specificity", "hedging", "formality", "structure"]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    # oracle reference (true attributes -> near-perfect ranking)
    ax.scatter(cov, orc, s=22, marker="_", c=C["grey"], alpha=0.7,
               label="oracle (true attributes)")
    # 20 random-combination criteria
    ax.scatter(cov[6:], jr[6:], s=45, c=C["neural"], alpha=0.75, edgecolor="w",
               linewidth=0.5, label="random-combination criteria (20)")
    # 6 pure attributes
    ax.scatter(cov[:6], jr[:6], s=95, c=C["dblind"], edgecolor="k", linewidth=0.6,
               zorder=5, label="interpretable attributes (6)")
    for i, name in enumerate(attr):
        dy = -0.055 if name in ("sentiment", "hedging", "length") else 0.03
        ax.annotate(name, (cov[i], jr[i]), (cov[i], jr[i] + dy), fontsize=8,
                    ha="center", color="k")
    # trend line
    b, a = np.polyfit(cov, jr, 1)
    xs = np.linspace(cov.min(), cov.max(), 50)
    ax.plot(xs, a + b * xs, "-", c=C["full"], lw=1.6, alpha=0.8)
    ax.set_xlabel("representation coverage of the criterion  (out-of-sample $R^2$)")
    ax.set_ylabel("judge's ranking reliability  (rank correlation)")
    ax.set_ylim(0.1, 1.05)
    ax.set_title("The coverage law: how well a frozen judge can READ a criterion predicts\n"
                 "how reliably it can RANK on it  (Qwen2.5-3B, 26 criteria)", fontsize=10)
    ax.text(0.02, 0.97,
            f"Spearman = +{spear:.2f}\nshuffled-label null $R^2\\approx$0 "
            f"(mean {null_mean:.2f})\noracle ranks near-perfect",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec=C["grey"], alpha=0.9))
    ax.axvspan(0, 0.30, color=C["dblind"], alpha=0.06)
    ax.text(0.15, 0.42, "low-coverage band\n(unreliable to select on)", ha="center",
            fontsize=8, color=C["dblind"])
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_coverage_law.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)


# ------------------------------------------------------------------ Fig 5
def fig5_correctness_drift():
    """Drift onto a REAL blind criterion: TruthfulQA factual correctness
    (is_correct coverage R^2=0.20). As the standard drifts length->correctness,
    the judge's rank on correctness degrades 0.81->0.35 while it still tracks the
    oracle in absolute top-k terms (a graded, not catastrophic, failure)."""
    J = json.load(open(SEM))
    ps = J["per_stage"]
    frac = [p["frac"] for p in ps]
    jrank = [p["judge_rank"] for p in ps]
    orc_c = [p["oracle_topk_correct"] for p in ps]
    jud_c = [p["judge_topk_correct"] for p in ps]
    r2 = [c for c in J["coverage"] if c["attr"] == "is_correct"][0]["ridge_r2"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    axA.plot(frac, jrank, "-o", c=C["dblind"], lw=2.2, label="3B judge")
    axA.axhline(1.0, ls="--", c=C["full"], lw=1.5, label="oracle (true attribute)")
    axA.annotate(f"+{jrank[0]:.2f}", (frac[0], jrank[0]), (0.04, 0.86), fontsize=9,
                 color=C["dblind"])
    axA.annotate(f"+{jrank[-1]:.2f}", (frac[-1], jrank[-1]), (0.80, 0.44), fontsize=9,
                 color=C["dblind"])
    axA.set_ylim(0.25, 1.05)
    axA.set_xlabel("standard drifts  length $\\rightarrow$ factual correctness  (fraction)")
    axA.set_ylabel("ranking reliability on correctness")
    axA.set_title("The judge loses the ability to\nRANK by correctness")
    axA.legend(fontsize=8, loc="lower left")

    axB.plot(frac, orc_c, "-s", c=C["full"], lw=1.8, label="oracle top-k")
    axB.plot(frac, jud_c, "-o", c=C["dblind"], lw=2.0, label="3B judge top-k")
    axB.axhline(J["base_rate"], ls=":", c=C["grey"], lw=1, label="chance (0.50)")
    axB.fill_between(frac, jud_c, orc_c, color=C["grey"], alpha=0.15)
    axB.set_ylim(0.4, 1.05)
    axB.set_xlabel("standard drift (fraction)")
    axB.set_ylabel("top-k answers actually correct")
    axB.set_title("...yet in absolute terms it improves —\na graded, not catastrophic, gap")
    axB.legend(fontsize=8, loc="lower right")

    fig.suptitle("Drift onto a real blind criterion — factual correctness on TruthfulQA "
                 f"(the 3B judge covers it only at $R^2$={r2:.2f})", fontsize=10.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_correctness_drift.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    fig1_dissociation()
    fig2_neural_transfer()
    fig3_exploitability()
    fig4_coverage_law()
    fig5_correctness_drift()
    print("done ->", OUT)
