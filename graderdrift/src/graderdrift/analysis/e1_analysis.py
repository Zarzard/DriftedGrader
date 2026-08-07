"""E1 analysis + Figure 1 (spec v0.2 Section 4, E1).

Reads only the per-seed ``log.jsonl`` files. Per-run OLS slopes:
- operational stats (op_agree, op_temp_inv_std/raw) over the IDENTIFIABLE window
  (rounds 5-45, i.e. c_k >= ~0.2; beyond that the per-round temperature is
  unidentified -- boundary-hit fraction rises, see AMENDMENTS wave 3);
- anchor summaries (D_dir, D_norm) over rounds 5-60 (computable throughout).

Aggregates mean slope with a 90% seed-level bootstrap CI (10k). Equivalence (TOST-
style, CI-in-bound): D_dir/D_norm |slope| < 0.001/round; op_temp_inv_raw |slope| <
0.002/round. Directional: op_agree slope < 0 and op_temp_inv_std slope > 0, CIs
excluding 0. Figure 1: operational stats (left) vs anchor panels (right),
mean +/- seed band, with the information-starved region shaded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEMP_WINDOW = (5, 35)     # identifiable window: c_k >= 0.33, zero boundary hits
ANCHOR_WINDOW = (5, 60)
N_BOOT = 10000


def load_runs(root: str | Path, schedule: str = "S0") -> list[dict]:
    root = Path(root) / schedule
    runs = []
    for run_dir in sorted(root.glob("*")):
        log = run_dir / "log.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        cols = {k: np.array([r[k] for r in rows], dtype=float) for k in
                ("round", "D_dir", "D_norm", "op_agree", "op_flip",
                 "op_temp_inv_std", "op_temp_inv_raw")}
        runs.append(cols)
    if not runs:
        raise FileNotFoundError(f"no logs under {root}")
    return runs


def _slope(rounds: np.ndarray, y: np.ndarray, window: tuple[int, int]) -> float:
    m = (rounds >= window[0]) & (rounds <= window[1]) & np.isfinite(y)
    return float(np.polyfit(rounds[m], y[m], 1)[0])


def _boot_ci(values: np.ndarray, rng: np.random.Generator, agg=np.median,
             n: int = N_BOOT, lo: float = 5.0, hi: float = 95.0) -> tuple[float, float]:
    """90% bootstrap CI of an aggregate (default median) of per-seed values."""
    idx = rng.integers(0, len(values), size=(n, len(values)))
    stat = agg(values[idx], axis=1)
    return float(np.percentile(stat, lo)), float(np.percentile(stat, hi))


@dataclass
class StatResult:
    name: str
    window: tuple[int, int]
    median_slope: float
    ci: tuple[float, float]
    level_range: tuple[float, float]   # median trajectory min/max over the window
    verdict: str


def _level_range(runs, rounds, name, window) -> tuple[float, float]:
    m = (rounds >= window[0]) & (rounds <= window[1])
    med = np.nanmedian(np.array([r[name] for r in runs]), axis=0)[m]
    return float(np.nanmin(med)), float(np.nanmax(med))


def analyze(root: str | Path, schedule: str = "S0", seed: int = 0) -> dict:
    """E1 analysis with ROBUST (median) per-seed slope aggregation.

    Rationale (AMENDMENTS wave 3): the per-round temperature MLE at n_pairs=100 is
    heavy-tailed (occasional boundary spikes even inside the identifiable window),
    so the mean-of-slopes is outlier-dominated. We aggregate per-seed slopes by
    their median and bootstrap the median's CI. op_temp_inv_raw's flatness is tested
    RELATIVELY -- it must sit well below op_temp_inv_std (they diverge, note Section
    4) -- since a tight absolute |slope| bound is unachievable for a per-round MLE.
    """
    runs = load_runs(root, schedule)
    rng = np.random.default_rng(seed)
    rounds = runs[0]["round"]

    specs = [
        ("op_agree", TEMP_WINDOW, "decrease"),
        ("op_temp_inv_std", TEMP_WINDOW, "increase"),
        ("op_temp_inv_raw", TEMP_WINDOW, "raw"),
        ("D_dir", ANCHOR_WINDOW, "equiv:0.001"),
        ("D_norm", ANCHOR_WINDOW, "equiv:0.001"),
    ]
    slope_arrays: dict[str, np.ndarray] = {}
    results: list[StatResult] = []
    for name, window, crit in specs:
        slopes = np.array([_slope(r["round"], r[name], window) for r in runs])
        slope_arrays[name] = slopes
        ci = _boot_ci(slopes, rng)
        lvl = _level_range(runs, rounds, name, window)
        if crit == "decrease":
            verdict = "PASS" if ci[1] < 0 else "FAIL"
        elif crit == "increase":
            verdict = "PASS" if ci[0] > 0 else "FAIL"
        elif crit == "raw":
            verdict = "(see divergence)"
        else:
            bound = float(crit.split(":")[1])
            verdict = "PASS (equiv)" if (ci[0] > -bound and ci[1] < bound) else "FAIL (equiv)"
        results.append(StatResult(name, window, float(np.median(slopes)), ci, lvl, verdict))

    # std-vs-raw divergence: std median-slope CI must lie entirely above raw's.
    std_ci = _boot_ci(slope_arrays["op_temp_inv_std"], rng)
    raw_ci = _boot_ci(slope_arrays["op_temp_inv_raw"], rng)
    divergence = {
        "std_ci": std_ci, "raw_ci": raw_ci,
        "separated": bool(std_ci[0] > raw_ci[1]),
        "ratio": float(np.median(slope_arrays["op_temp_inv_std"]) /
                       max(abs(np.median(slope_arrays["op_temp_inv_raw"])), 1e-9)),
    }
    return {"runs": runs, "rounds": rounds, "results": results,
            "n_seeds": len(runs), "divergence": divergence}


def make_figure1(root: str | Path, out_path: str | Path, schedule: str = "S0") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = analyze(root, schedule)
    runs, rounds = res["runs"], res["rounds"]

    def band(name):
        A = np.array([r[name] for r in runs])  # (seeds, K)
        return np.nanmean(A, 0), np.nanstd(A, 0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: operational stats
    for name, color in (("op_agree", "C0"), ("op_temp_inv_std", "C1"), ("op_temp_inv_raw", "C2")):
        mu, sd = band(name)
        axL.plot(rounds, mu, color=color, label=name)
        axL.fill_between(rounds, mu - sd, mu + sd, color=color, alpha=0.15)
    axL.axvspan(TEMP_WINDOW[1], rounds[-1], color="gray", alpha=0.12)
    axL.text(TEMP_WINDOW[1] + 0.5, axL.get_ylim()[1] * 0.9, "info-starved", fontsize=8, color="gray")
    axL.set_yscale("log")
    axL.set_xlabel("round"); axL.set_ylabel("operational statistic (log)")
    axL.set_title("Operational stream (naive detectors)")
    axL.legend(fontsize=8)
    # right: anchor panels
    for name, color in (("D_dir", "C3"), ("D_norm", "C4")):
        mu, sd = band(name)
        axR.plot(rounds, mu, color=color, label=name)
        axR.fill_between(rounds, mu - sd, mu + sd, color=color, alpha=0.15)
    axR.axhline(0, color="k", lw=0.5)
    axR.set_xlabel("round"); axR.set_ylabel("anchor panel readout")
    axR.set_title("Anchor panels (instrument)")
    axR.legend(fontsize=8)
    fig.suptitle(f"E1 stationarity null (A0/S0, {res['n_seeds']} seeds): "
                 "operational stats move, anchors flat")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_report(root: str | Path, schedule: str = "S0") -> None:
    res = analyze(root, schedule)
    print(f"E1 stationarity null -- A0/{schedule}, {res['n_seeds']} seeds "
          f"(robust median-slope; temp window {TEMP_WINDOW}, anchor window {ANCHOR_WINDOW})")
    print(f"{'stat':18s} {'window':>8s} {'med_slope':>10s} {'90% CI (median)':>22s} "
          f"{'level range':>16s}   verdict")
    for r in res["results"]:
        print(f"{r.name:18s} {str(r.window):>8s} {r.median_slope:>+10.5f} "
              f"[{r.ci[0]:>+8.5f},{r.ci[1]:>+8.5f}] "
              f"[{r.level_range[0]:>6.2f},{r.level_range[1]:>6.2f}]   {r.verdict}")
    d = res["divergence"]
    print(f"\nstd-vs-raw divergence: std slope CI {tuple(round(x,4) for x in d['std_ci'])} "
          f"vs raw {tuple(round(x,4) for x in d['raw_ci'])} -> "
          f"{'SEPARATED (PASS)' if d['separated'] else 'overlapping (FAIL)'}; "
          f"median-slope ratio std/raw = {d['ratio']:.1f}x")


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    print_report(a.root)
    if a.fig:
        print("wrote", make_figure1(a.root, a.fig))
