"""M1 driver: drift-trajectory plots for every A0 preset (spec v0.2 M1 DoD).

For each A0 preset, run K=60 rounds from a fixed seed and record, per round:
    ||v||, angle(vhat, vhat_0) in degrees, realized rho_step and rho_cum
(against the A0 achievable rowspace P_{S_hist}). Emits a small-multiples PNG for
human review. Light enough for the login node (Agg backend, tiny numpy).

A1-only presets (S7*/S8*/SC*/SN) are not plotted here -- they need the C8/M6
achievable-gap geometry; their event functions are unit-tested in M1.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import drift
from ..geometry import rho, subspace_projector
from ..seeding import spawn_generators
from ..teacher import Population, Teacher


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cu = u / np.linalg.norm(u)
    cv = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(cu @ cv, -1.0, 1.0))))


def run_preset_trajectory(name: str, seed: int = 0, K: int = 60):
    """Return per-round arrays (norm, angle_deg, rho_step, rho_cum) for an A0 preset."""
    ctx = drift.DriftContext.a0(d=24, hist_dim=16)
    P_hist = subspace_projector(ctx.hist_idx, ctx.d)
    rngs = spawn_generators(seed)

    w0 = np.zeros(ctx.d)
    w0[ctx.hist_idx] = rngs["teacher"].standard_normal(len(ctx.hist_idx))
    w0 /= np.linalg.norm(w0)
    sched = drift.get_preset(name)

    is_pop = any(ev.type == "mixture_redraw" for ev in sched.events)
    norms, angles, rho_step, rho_cum = [], [], [], []

    if is_pop:
        pop = Population([Teacher(w=w0.copy(), beta=1.0) for _ in range(10)])
        v_prev = pop.v_mean()
        v0 = v_prev.copy()
        for k in range(1, K + 1):
            for ev in sched.events:
                if ev.t == k:
                    drift.mixture_redraw(pop, ctx, rngs["drift"], ev.params["fraction"])
            v = pop.v_mean()
            norms.append(np.linalg.norm(v))
            angles.append(_angle_deg(v, v0))
            rho_step.append(rho(v - v_prev, P_hist))
            rho_cum.append(rho(v - v0, P_hist))
            v_prev = v.copy()
    else:
        v = 1.0 * w0  # beta=1 => v0 = w0, unit
        v0 = v.copy()
        v_prev = v.copy()
        for k in range(1, K + 1):
            v, _flags = sched.apply_round(k, v, ctx, rngs["drift"])
            norms.append(np.linalg.norm(v))
            angles.append(_angle_deg(v, v0))
            rho_step.append(rho(v - v_prev, P_hist))
            rho_cum.append(rho(v - v0, P_hist))
            v_prev = v.copy()

    return (np.array(norms), np.array(angles), np.array(rho_step), np.array(rho_cum))


def make_figure(out_path: str | Path, seed: int = 0, K: int = 60) -> Path:
    presets = drift.A0_PRESETS
    ncol = 3
    nrow = int(np.ceil(len(presets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), squeeze=False)
    rounds = np.arange(1, K + 1)
    for i, name in enumerate(presets):
        ax = axes[i // ncol][i % ncol]
        norms, angles, _rs, rc = run_preset_trajectory(name, seed=seed, K=K)
        ax.plot(rounds, angles, color="C0", label="angle(v,v0) deg")
        ax.plot(rounds, rc * 90.0, color="C2", ls="--", label="rho_cum (x90)")
        ax.set_ylim(-2, 95)
        ax2 = ax.twinx()
        ax2.plot(rounds, norms, color="C3", label="||v||")
        ax2.set_ylim(bottom=0)
        ax.set_title(name)
        ax.set_xlabel("round")
        if i % ncol == 0:
            ax.set_ylabel("deg  /  rho_cum*90")
        if i % ncol == ncol - 1:
            ax2.set_ylabel("||v||", color="C3")
    for j in range(len(presets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # one shared legend
    handles = [
        plt.Line2D([], [], color="C0", label="angle(v, v0) [deg]"),
        plt.Line2D([], [], color="C2", ls="--", label="rho_cum x90"),
        plt.Line2D([], [], color="C3", label="||v||"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"A0 drift-preset trajectories (seed={seed}, K={K})", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":  # pragma: no cover
    p = make_figure(Path(__file__).resolve().parents[3] / "artifacts" / "m1_trajectories.png")
    print(f"wrote {p}")
