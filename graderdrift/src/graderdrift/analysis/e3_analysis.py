"""E3 analysis: estimator calibration (spec v0.2 Section 4, E3).

Reads only the per-run ``log.jsonl``. Produces:
- a coverage table: on-support (P_A v) per-direction 90% CI coverage, norm/angle
  interval coverage, mean on-support NEES, per (mode, schedule);
- the S0 noise floor for the drift statistics (net + path-length), written to
  ``analysis/noise_floor.json`` for E4;
- the trajectory figure: filtered ||v|| and angle vs truth with 90% bands (S1, S5).

Gate (known_calendar): on-support coverage in [85, 95] pooled and per-schedule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import estimator as est

MODES = ["informed", "known_calendar", "blind"]      # informed = C5 (generator-matched Q)
SCHEDULES = ["S0", "S1", "S2", "S3", "S4", "S5"]     # S3 = population, report-only (no gate)
GATED_MODE = "informed"                               # pending C5 sign-off at M5 review
GATED_SCHEDULES = ["S0", "S1", "S2", "S4", "S5"]     # S3 excluded (misspecified by design)
Z90 = 1.645


def load_e3_run(run_dir: Path) -> dict:
    rows = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines() if l.strip()]
    d = rows[0]
    crit = lambda r, k: r["D_crit"][k]
    return {
        "round": np.array([r["round"] for r in rows]),
        "m_k": np.array([r["m_k"] for r in rows]),
        "v_true": np.array([r["v_true"] for r in rows]),
        "os_true": np.array([crit(r, "onsupport_true") for r in rows]),
        "os_mean": np.array([crit(r, "onsupport_mean") for r in rows]),
        "os_sd": np.array([crit(r, "onsupport_sd") for r in rows]),
        "nees_on": np.array([crit(r, "nees_on") for r in rows]),
        "norm_true": np.array([crit(r, "norm_true") for r in rows]),
        "norm_lo": np.array([crit(r, "norm_lo") for r in rows]),
        "norm_hi": np.array([crit(r, "norm_hi") for r in rows]),
        "angle_true": np.array([crit(r, "angle_true") for r in rows]),
        "angle_lo": np.array([crit(r, "angle_lo") for r in rows]),
        "angle_hi": np.array([crit(r, "angle_hi") for r in rows]),
    }


def load_cell(root, mode, schedule) -> list[dict]:
    cell = Path(root) / mode / schedule
    return [load_e3_run(d) for d in sorted(cell.glob("*")) if (d / "log.jsonl").exists()]


def cell_coverage(runs: list[dict]) -> dict:
    os_hit, norm_hit, ang_hit, nees = [], [], [], []
    for r in runs:
        os_hit.append(np.abs(r["os_true"] - r["os_mean"]) <= Z90 * r["os_sd"])
        norm_hit.append((r["norm_true"] >= r["norm_lo"]) & (r["norm_true"] <= r["norm_hi"]))
        ang_hit.append((r["angle_true"] >= r["angle_lo"]) & (r["angle_true"] <= r["angle_hi"]))
        nees.append(r["nees_on"])
    return {
        "onsupport_cov": float(np.mean(np.concatenate([h.ravel() for h in os_hit]))),
        "norm_cov": float(np.mean(np.concatenate(norm_hit))),
        "angle_cov": float(np.mean(np.concatenate(ang_hit))),
        "nees_on_mean": float(np.mean(np.concatenate(nees))),
        "n_runs": len(runs),
    }


def coverage_table(root) -> dict:
    table = {}
    for mode in MODES:
        for sched in SCHEDULES:
            runs = load_cell(root, mode, sched)
            if runs:
                table[(mode, sched)] = cell_coverage(runs)
    return table


def print_coverage_table(root) -> None:
    table = coverage_table(root)
    dof = 16
    print(f"E3 estimator calibration (on-support dof={dof}; 90% CIs; gate [85,95] on "
          f"{GATED_MODE}; S3 report-only)")
    print(f"{'mode':16s} {'sched':5s} {'P_A cov':>8s} {'norm cov':>9s} {'angle cov':>10s} "
          f"{'NEES/dof':>9s} {'gate':>6s}")
    for mode in MODES:
        for sched in SCHEDULES:
            c = table.get((mode, sched))
            if not c:
                continue
            gate = ""
            if mode == GATED_MODE and sched in GATED_SCHEDULES:
                gate = "PASS" if 0.85 <= c["onsupport_cov"] <= 0.95 else "FAIL"
            elif sched == "S3":
                gate = "(rep)"
            print(f"{mode:16s} {sched:5s} {c['onsupport_cov']:>8.3f} {c['norm_cov']:>9.3f} "
                  f"{c['angle_cov']:>10.3f} {c['nees_on_mean']/dof:>9.2f} {gate:>6s}")
    # pooled gated-mode on-support coverage (gated schedules only)
    pooled = [load_cell(root, GATED_MODE, s) for s in GATED_SCHEDULES]
    all_runs = [r for cell in pooled for r in cell]
    if all_runs:
        pc = cell_coverage(all_runs)["onsupport_cov"]
        print(f"\n{GATED_MODE} POOLED on-support coverage ({'+'.join(GATED_SCHEDULES)}): {pc:.3f} "
              f"-> {'PASS' if 0.85 <= pc <= 0.95 else 'FAIL'}")


def compute_noise_floor(root, out_path=None, mode=GATED_MODE) -> dict:
    """S0 noise floor for the drift statistics (for E4), from the gated mode."""
    runs = load_cell(root, mode, "S0")
    tv_full, net_disp, net_ln, net_ang = [], [], [], []
    for r in runs:
        M = r["m_k"]
        tv_full.append(est.total_variation(M))
        net_disp.append(est.net_displacement(M))
        net_ln.append(est.net_lognorm_change(M))
        net_ang.append(est.net_angle_change(M))
    floor = {
        "tv_full": {"mean": float(np.mean(tv_full)), "sd": float(np.std(tv_full))},
        "net_disp": {"mean": float(np.mean(net_disp)), "sd": float(np.std(net_disp))},
        "net_lognorm": {"mean": float(np.mean(net_ln)), "sd": float(np.std(net_ln))},
        "net_angle": {"mean": float(np.mean(net_ang)), "sd": float(np.std(net_ang))},
        "n_runs": len(runs),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(floor, indent=2))
    return floor


def make_e3_figure(root, out_path, schedules=("S1", "S5"), mode="known_calendar", seed=0) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(schedules), 2, figsize=(11, 3.4 * len(schedules)), squeeze=False)
    for i, sched in enumerate(schedules):
        runs = load_cell(root, mode, sched)
        if not runs:
            continue
        r = runs[min(seed, len(runs) - 1)]
        rounds = r["round"]
        M = r["m_k"]
        norm_est = np.linalg.norm(M, axis=1)
        m0 = M[0]
        ang_est = np.degrees(np.arccos(np.clip((M @ m0) / (norm_est * np.linalg.norm(m0)), -1, 1)))
        axN = axes[i][0]
        axN.plot(rounds, r["norm_true"], "k-", label="||v|| true")
        axN.plot(rounds, norm_est, "C0-", label="||v|| est")
        axN.fill_between(rounds, r["norm_lo"], r["norm_hi"], color="C0", alpha=0.2, label="90% band")
        axN.set_title(f"{sched}: norm"); axN.set_xlabel("round"); axN.set_ylabel("||v||")
        axN.legend(fontsize=8)
        axA = axes[i][1]
        axA.plot(rounds, r["angle_true"], "k-", label="angle true")
        axA.plot(rounds, ang_est, "C1-", label="angle est")
        axA.fill_between(rounds, r["angle_lo"], r["angle_hi"], color="C1", alpha=0.2, label="90% band")
        axA.set_title(f"{sched}: angle(vhat, vhat_0)"); axA.set_xlabel("round"); axA.set_ylabel("deg")
        axA.legend(fontsize=8)
    fig.suptitle(f"E3 filtered trajectories vs truth ({mode}, seed {seed})")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--floor", default=None)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    print_coverage_table(a.root)
    if a.floor:
        print("\nnoise floor:", json.dumps(compute_noise_floor(a.root, a.floor), indent=2))
    if a.fig:
        print("wrote", make_e3_figure(a.root, a.fig))
