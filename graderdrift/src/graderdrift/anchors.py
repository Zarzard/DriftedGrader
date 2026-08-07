"""C4. Anchor battery (spec v0.2 Section 3, C4).

Built once at round 0 from the round-0 gap distribution, then FROZEN (bytes
identical across the run). Two panels (note Section 5c):

- **Direction panel** (~50/50, `|p-0.5| <= 0.05`): greedy D-optimal on the Fisher
  information `sigma'(z) dphi dphi^T` -- maximally sensitive to direction drift,
  spread out so no rotation hides in the null space.
- **Norm panel** (~80/20, `p in [0.78,0.86] u [0.14,0.22]`): `|z|` nearest z* =
  1.5434, sign-balanced -- maximally sensitive to scale drift.

A0 candidate pool: `N(0, c0^2 I)` on S_hist, exactly zero on S_nov, so
rowspace(Phi_A) subset S_hist and rank(Phi_A) = 16. The A1 battery (segment pairs +
criterion-contrastive panel) is built in C8/M6 and reuses the panel selectors here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._math import Z_STAR, sigmoid, sigmoid_prime
from .drift import DriftContext
from .geometry import projector_from_rowspace

N_CAND_DEFAULT = 5000
M_DIR_A0 = 16
M_NORM_A0 = 8


@dataclass(frozen=True)
class AnchorBattery:
    """Frozen anchor gaps and panel bookkeeping.

    gaps: (m, d) feature gaps (immutable via read-only view).
    z0: v0 . gap_j at build time. dir_idx / norm_idx index into ``gaps``.
    P_A: projector onto rowspace(gaps). rank: rank(gaps).
    """

    gaps: np.ndarray
    z0: np.ndarray
    dir_idx: np.ndarray
    norm_idx: np.ndarray
    P_A: np.ndarray
    rank: int
    min_pairwise_angle_deg: float

    def p0_true(self) -> np.ndarray:
        """True round-0 label probabilities sigma(z0) (for diagnostics/tests)."""
        return sigmoid(self.z0)


def _min_pairwise_angle_deg(gaps: np.ndarray) -> float:
    g = gaps / np.linalg.norm(gaps, axis=1, keepdims=True)
    c = np.clip(g @ g.T, -1.0, 1.0)
    np.fill_diagonal(c, 0.0)
    return float(np.degrees(np.arccos(c.max())))


def _greedy_d_optimal(cand_gaps: np.ndarray, cand_z: np.ndarray, m: int, d: int,
                      ridge: float = 1e-6) -> np.ndarray:
    """Greedy D-optimal selection of ``m`` gaps maximizing log det of accumulated
    Fisher information. Returns indices into ``cand_gaps``.

    Uses the matrix-determinant lemma: the log-det increment from adding gap g with
    weight w = sigma'(z) is log(1 + w * g^T A^{-1} g).
    """
    A = ridge * np.eye(d)
    A_inv = np.linalg.inv(A)
    weights = sigmoid_prime(cand_z)
    chosen: list[int] = []
    available = np.ones(len(cand_gaps), dtype=bool)
    for _ in range(m):
        # score = log(1 + w_j g_j^T A_inv g_j) for each available candidate
        Ag = cand_gaps @ A_inv  # (N, d)
        quad = np.einsum("ij,ij->i", Ag, cand_gaps)  # g^T A_inv g
        score = np.log1p(weights * quad)
        score[~available] = -np.inf
        j = int(np.argmax(score))
        chosen.append(j)
        available[j] = False
        # rank-1 update of A_inv (Sherman-Morrison)
        g = cand_gaps[j]
        w = weights[j]
        Ainv_g = A_inv @ g
        A_inv = A_inv - w * np.outer(Ainv_g, Ainv_g) / (1.0 + w * (g @ Ainv_g))
    return np.array(chosen, dtype=int)


def _select_norm_panel(cand_gaps: np.ndarray, cand_z: np.ndarray, m_norm: int) -> np.ndarray:
    """Pick ``m_norm`` gaps with |z| nearest z*, balanced across sign."""
    half = m_norm // 2
    pos = np.where(cand_z > 0)[0]
    neg = np.where(cand_z < 0)[0]
    pos = pos[np.argsort(np.abs(np.abs(cand_z[pos]) - Z_STAR))][:half]
    neg = neg[np.argsort(np.abs(np.abs(cand_z[neg]) - Z_STAR))][:half]
    return np.concatenate([pos, neg])


# A1 criterion contrasts (spec v0.2 C4): name -> unit contrast vector in R^8.
# Contrasts are DIFFERENCES by design -- individual cell-type levels lie partly in
# null(S_ach) and are not separately identifiable (amendment B5).
def a1_contrasts() -> dict[str, np.ndarray]:
    c = {}
    def unit(vec):
        v = np.zeros(8);
        for i, x in vec: v[i] = x
        return v / np.linalg.norm(v)
    c["lava_vs_grass"] = unit([(2, 1.0), (4, -1.0)])
    c["water_vs_grass"] = unit([(3, 1.0), (4, -1.0)])
    c["wall_bump"] = unit([(5, 1.0)])
    c["stay_action"] = unit([(6, 1.0)])
    c["goal_proximity"] = unit([(1, 1.0)])
    return c


@dataclass(frozen=True)
class A1Battery:
    """Frozen A1 battery: direction + norm panels plus the criterion-contrastive
    panel (per-contrast anchor indices, signs, and purities)."""

    gaps: np.ndarray
    z0: np.ndarray
    dir_idx: np.ndarray
    norm_idx: np.ndarray
    contrast_idx: dict          # name -> np.ndarray of row indices into gaps
    contrast_sign: dict         # name -> +/-1 per anchor (sign of c . gap)
    contrast_purity: dict       # name -> mean purity
    purity_flags: dict          # name -> bool (mean purity < 0.6 after knobs)
    P_A: np.ndarray
    rank: int


def build_battery_a1(v0: np.ndarray, cand_gaps: np.ndarray, rng: np.random.Generator,
                     m_dir: int = 12, m_norm: int = 6, m_c: int = 4,
                     purity_min: float = 0.6) -> A1Battery:
    """Build the frozen A1 battery from round-0 reference segment-pair gaps.

    Panels (spec v0.2 C4): direction m_dir (|p-0.5|<=0.05, greedy D-optimal), norm
    m_norm (bands, |z| nearest z*, sign-balanced), plus m_c anchors per criterion
    contrast maximizing purity |c . gap|/||gap|| within p in [0.25, 0.75]
    (sanctioned knob: widen to [0.20, 0.80] if mean purity < 0.6). rank <= 6.
    """
    cand = np.asarray(cand_gaps, dtype=float)
    z = cand @ np.asarray(v0, dtype=float)
    p = sigmoid(z)
    d = cand.shape[1]

    dir_pool = np.where(np.abs(p - 0.5) <= 0.05)[0]
    if len(dir_pool) < m_dir:
        raise RuntimeError(f"A1 direction pool too small ({len(dir_pool)}); enlarge the pair pool")
    dir_sel = dir_pool[_greedy_d_optimal(cand[dir_pool], z[dir_pool], m_dir, d)]

    norm_pool = np.where(((p >= 0.78) & (p <= 0.86)) | ((p >= 0.14) & (p <= 0.22)))[0]
    if len(norm_pool) < m_norm:
        raise RuntimeError(f"A1 norm pool too small ({len(norm_pool)}); enlarge the pair pool")
    norm_sel = norm_pool[_select_norm_panel(cand[norm_pool], z[norm_pool], m_norm)]

    rows = [cand[dir_sel], cand[norm_sel]]
    contrast_idx, contrast_sign, contrast_purity, purity_flags = {}, {}, {}, {}
    offset = m_dir + m_norm
    gap_norms = np.linalg.norm(cand, axis=1)
    taken = set(dir_sel.tolist()) | set(norm_sel.tolist())
    for name, cvec in a1_contrasts().items():
        purity_all = np.abs(cand @ cvec) / np.maximum(gap_norms, 1e-12)
        chosen = None
        for lo, hi in ((0.25, 0.75), (0.20, 0.80)):        # sanctioned widen knob
            pool = np.where((p >= lo) & (p <= hi))[0]
            pool = np.array([i for i in pool if i not in taken])
            if len(pool) >= m_c:
                order = pool[np.argsort(-purity_all[pool])][:m_c]
                if purity_all[order].mean() >= purity_min or (lo, hi) == (0.20, 0.80):
                    chosen = order
                    break
        if chosen is None:                                  # reduce m_c knob
            chosen = pool[np.argsort(-purity_all[pool])][:max(len(pool), 1)]
        taken |= set(chosen.tolist())
        contrast_idx[name] = np.arange(offset, offset + len(chosen))
        contrast_sign[name] = np.sign(cand[chosen] @ cvec)
        contrast_purity[name] = float(purity_all[chosen].mean())
        purity_flags[name] = bool(contrast_purity[name] < purity_min)
        rows.append(cand[chosen])
        offset += len(chosen)

    gaps = np.vstack(rows)
    gaps.setflags(write=False)
    P_A = projector_from_rowspace(gaps, tol=1e-8 * np.linalg.svd(gaps, compute_uv=False)[0])
    rank = int(np.linalg.matrix_rank(gaps, tol=1e-6))
    if rank > 6:
        raise AssertionError(f"A1 battery rank {rank} > 6 -- feature null space violated")
    return A1Battery(gaps=gaps, z0=gaps @ v0, dir_idx=np.arange(m_dir),
                     norm_idx=np.arange(m_dir, m_dir + m_norm),
                     contrast_idx=contrast_idx, contrast_sign=contrast_sign,
                     contrast_purity=contrast_purity, purity_flags=purity_flags,
                     P_A=P_A, rank=rank)


def build_battery_a0(v0: np.ndarray, ctx: DriftContext, rng: np.random.Generator,
                     n_cand: int = N_CAND_DEFAULT, m_dir: int = M_DIR_A0,
                     m_norm: int = M_NORM_A0, c0: float = 2.0,
                     min_angle_deg: float = 5.0, max_resamples: int = 4) -> AnchorBattery:
    """Build the frozen A0 anchor battery against ``v0`` (unit vector in S_hist)."""
    d = ctx.d
    for attempt in range(max_resamples + 1):
        n = n_cand * (attempt + 1)  # grow pool if degeneracy detected
        cand = np.zeros((n, d))
        cand[:, ctx.hist_idx] = rng.standard_normal((n, len(ctx.hist_idx))) * c0
        z = cand @ v0
        p = sigmoid(z)

        dir_pool = np.where(np.abs(p - 0.5) <= 0.05)[0]
        norm_pool = np.where(((p >= 0.78) & (p <= 0.86)) | ((p >= 0.14) & (p <= 0.22)))[0]
        if len(dir_pool) < m_dir or len(norm_pool) < m_norm:
            continue

        dir_sel_local = _greedy_d_optimal(cand[dir_pool], z[dir_pool], m_dir, d)
        dir_sel = dir_pool[dir_sel_local]
        norm_sel_local = _select_norm_panel(cand[norm_pool], z[norm_pool], m_norm)
        norm_sel = norm_pool[norm_sel_local]

        gaps = np.vstack([cand[dir_sel], cand[norm_sel]])
        min_angle = _min_pairwise_angle_deg(gaps)
        if min_angle <= min_angle_deg:
            continue  # degenerate; resample with a larger pool

        gaps.setflags(write=False)
        P_A = projector_from_rowspace(gaps)
        rank = int(np.linalg.matrix_rank(gaps, tol=1e-8))
        return AnchorBattery(
            gaps=gaps,
            z0=gaps @ v0,
            dir_idx=np.arange(m_dir),
            norm_idx=np.arange(m_dir, m_dir + m_norm),
            P_A=P_A,
            rank=rank,
            min_pairwise_angle_deg=min_angle,
        )
    raise RuntimeError("failed to build a nondegenerate battery; increase N_cand")
