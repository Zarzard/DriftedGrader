"""Test 8 (spec v0.2 Section 6): temperature aliasing under S0.

With a constant rater and shrinking gap scale c_k, the STANDARDIZED-margin
temperature 1/tau_std rises ~ 1/c_k (the aliasing statistic dashboards watch),
while the RAW-margin temperature is identified (~||v||) and stays flat. This
isolates the C6 statistic's population behavior using the true direction and a
large per-round sample (so the estimator variance is small); the E1 experiment
then shows it at the specced n_pairs=100 with the identifiable-window caveat.
"""
import numpy as np

from graderdrift.diagnostics import op_temp_inv
from graderdrift.streams import gap_scale


def _round_labels(k, v0, rng, n=2000, d=24, hist=16):
    c_k = gap_scale(k)
    gaps = np.zeros((n, d))
    gaps[:, :hist] = rng.standard_normal((n, hist)) * c_k
    p = 1.0 / (1.0 + np.exp(-(gaps @ v0)))
    labels = (rng.random(n) < p).astype(int)
    return gaps, labels


def test_std_rises_raw_flat():
    rng = np.random.default_rng(0)
    v0 = np.zeros(24); v0[:16] = rng.standard_normal(16); v0 /= np.linalg.norm(v0)
    ks = np.arange(1, 41)  # informative window: c_k in [1.55, 0.26]
    inv_c = np.array([1.0 / gap_scale(k) for k in ks])
    std, raw = [], []
    for k in ks:
        gaps, labels = _round_labels(k, v0, rng)
        s, r = op_temp_inv(gaps, labels, v0)
        std.append(s); raw.append(r)
    std = np.array(std); raw = np.array(raw)

    # 1/tau_std ~ linear in 1/c_k with high R^2
    A = np.polyfit(inv_c, std, 1)
    pred = np.polyval(A, inv_c)
    ss_res = np.sum((std - pred) ** 2)
    ss_tot = np.sum((std - std.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert A[0] > 0, "std temperature must rise with 1/c_k"
    assert r2 > 0.9, f"std vs 1/c_k R^2={r2:.3f}"

    # raw temperature flat: its slope on 1/c_k is tiny vs the std slope
    B = np.polyfit(inv_c, raw, 1)
    assert abs(B[0]) < 0.1 * abs(A[0]), f"raw slope {B[0]:.3f} not flat vs std slope {A[0]:.3f}"
    assert 0.5 < np.median(raw) < 2.0, f"raw median {np.median(raw):.2f} should sit near ||v||=1"
