"""Stage A driver: generator characterisation + mechanism (VR) phase diagram.

Stage A has no reward model (RM-free): the agent tracks w_{t-L}+noise and plays
Gibbs. It does NOT establish the phenomenon (that is Stage B); it maps the drift
dynamics and the VR mechanism, and establishes the regret floor.

Produces two figures from the parquet logs:
  Fig 1  : (lambda, beta) VR phase diagram, one panel per drift model.
  Fig 1b : the R_bar regret floor vs (L, sigma), flat in t.

Usage:  python -m analysis.run_stage_a --seeds 10 --workers 8
Dev seeds by default (X1: fix design on dev; report seeds run once at freeze).
"""
from __future__ import annotations

import argparse

from drifted_grader.runner import Config
from drifted_grader.sweeps import grid_configs, run_grid, RUNS_ROOT
from analysis import figures


DEV_SEEDS = list(range(10))
REPORT_SEEDS = list(range(100, 120))


def build_configs():
    # Phase-diagram cells (clean driver: L=0, sigma=0), VR mechanism + drift geometry
    base = Config(stage="A", rm="none", T=100, schema="reduced", L=0, sigma=0.0)
    phase = grid_configs(
        base,
        drift=["SAT", "EXO", "REPLAY", "SCALE"],
        lam=[0.05, 0.1, 0.2, 0.4, 0.8],
        beta=[0.02, 0.05, 0.1, 0.2, 0.5],
    )
    # NONE only needs lam=0 (drift ignores lam) -> add a small set
    none_cells = grid_configs(base, drift=["NONE"], lam=[0.0],
                              beta=[0.02, 0.05, 0.1, 0.2, 0.5])
    # Regret-floor cells (Fig 1b): vary L, sigma at a couple (lam, beta)
    floor = grid_configs(
        Config(stage="A", rm="none", T=100, schema="reduced", drift="SAT"),
        lam=[0.1, 0.4], beta=[0.05], L=[0, 1, 2, 4], sigma=[0.0, 0.05, 0.1],
    )
    return phase + none_cells + floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--report", action="store_true", help="use report seeds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--figdir", default="figs")
    args = ap.parse_args()

    seeds = REPORT_SEEDS[:args.seeds] if args.report else DEV_SEEDS[:args.seeds]
    configs = build_configs()
    print(f"Stage A: {len(configs)} cells x {len(seeds)} seeds "
          f"= {len(configs)*len(seeds)} runs")
    paths, crashes = run_grid(configs, seeds, n_workers=args.workers)
    print(f"wrote {len(paths)} run files; {len(crashes)} crashes")
    for c in crashes:
        print("  CRASH:", c)

    # figures from the phase-diagram cells
    df = figures.load_runs(f"{RUNS_ROOT}/A/**/*.parquet")
    phase_df = df[df["drift_model"].isin(["SAT", "EXO", "REPLAY", "SCALE", "NONE"])]
    figures.fig1_phase_diagram(phase_df, f"{args.figdir}/fig1_vr_phase.png")
    floor_df = df[(df["L"] > 0) | (df["sigma"] > 0) | (df["drift_model"] == "SAT")]
    figures.fig1b_regret_floor(floor_df, f"{args.figdir}/fig1b_regret_floor.png")
    print(f"figures written to {args.figdir}/")


if __name__ == "__main__":
    main()
