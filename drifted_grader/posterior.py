"""Reward posterior over linear reward w (Plan 02, Section 7).

P(w | D) ∝ N(0, prior_scale^2 I) · Π_{(i>j)} sigmoid( betaT · w·(φ_i - φ_j) )

The likelihood depends on w only through dΦ @ w, so a BATCH of candidate w's is a
single matmul (Section 7.3). Sampler = Laplace proposal + importance reweighting:

  1. MAP by L-BFGS (BT posterior with Gaussian prior is log-concave -> reliable).
  2. Gaussian proposal q = N(MAP, H^{-1}), H the Hessian at the MAP.
  3. Draw M from q, log-weights = log p(w|D) - log q(w) in one batched matmul.
  4. Weighted quantiles for the VaR lower bound; log ESS.

CRITICAL (units): fit with the KNOWN betaT so posterior w is on the SAME scale as
the grader's w_t. Otherwise coverage compares βT·w to w and fails trivially.
Do NOT use the Laplace mean alone for the tail quantile -- reweight (the Gaussian
approximation is worst exactly in the tail, which is the whole quantity of interest).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _softplus(x):
    return np.logaddexp(0.0, x)


def weighted_quantile(values, weights, q):
    """Weighted q-quantile (q in [0,1]); linear interpolation on the weighted CDF."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    # midpoint convention for stability
    cw_mid = cw - 0.5 * w / w.sum()
    return float(np.interp(q, cw_mid, v))


class RewardPosterior:
    def __init__(self, k: int):
        self.k = int(k)
        self.map_w = np.zeros(self.k)
        self.samples = None
        self.weights = None
        self.cov = np.eye(self.k)
        self.ess = float("nan")
        self.betaT = None

    # ---- fit: MAP + Laplace + importance reweighting -------------------
    def fit(self, dPhi, y, betaT, prior_scale=1.0, n_samples=10000, rng=None,
            chunk=2000):
        dPhi = np.asarray(dPhi, dtype=float)
        y = np.asarray(y, dtype=float)
        self.betaT = float(betaT)
        s2 = float(prior_scale) ** 2
        M, k = dPhi.shape
        assert k == self.k

        def nll_grad(w):
            z = self.betaT * (dPhi @ w)
            p = expit(z)
            nll = np.sum(y * _softplus(-z) + (1 - y) * _softplus(z)) + 0.5 * (w @ w) / s2
            g = self.betaT * (dPhi.T @ (p - y)) + w / s2
            return nll, g

        res = minimize(nll_grad, self.map_w, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.map_w = res.x

        # Hessian at the MAP: betaT^2 dΦ^T diag(p(1-p)) dΦ + I/s2
        z = self.betaT * (dPhi @ self.map_w)
        pw = expit(z) * (1 - expit(z))
        H = (self.betaT ** 2) * (dPhi.T * pw) @ dPhi + np.eye(k) / s2
        self.cov = np.linalg.inv(H)

        rng = rng or np.random.default_rng(0)
        Lc = np.linalg.cholesky(self.cov)
        eps = rng.standard_normal((n_samples, k))
        samples = self.map_w + eps @ Lc.T                # (n_samples, k)

        dm = samples - self.map_w
        logq = -0.5 * np.einsum("ij,jk,ik->i", dm, H, dm)

        # batched log-posterior over samples (chunk to bound memory)
        logpost = np.empty(n_samples)
        for s0 in range(0, n_samples, chunk):
            sl = slice(s0, s0 + chunk)
            Z = self.betaT * (samples[sl] @ dPhi.T)      # (chunk, M)
            nll = (y[None, :] * _softplus(-Z) + (1 - y[None, :]) * _softplus(Z)).sum(1)
            nll += 0.5 * (samples[sl] ** 2).sum(1) / s2
            logpost[sl] = -nll

        logw = logpost - logq
        logw -= logw.max()
        wts = np.exp(logw)
        wts /= wts.sum()
        self.samples = samples
        self.weights = wts
        self.ess = float(1.0 / np.sum(wts ** 2))
        return self

    # ---- posterior queries --------------------------------------------
    def value_dist(self, f_pi):
        """{ w_i · f_pi } over posterior samples (weights in self.weights)."""
        return self.samples @ np.asarray(f_pi, dtype=float)

    def lower_bound(self, f_pi, alpha):
        """VaR: the alpha-quantile (lower tail) of the value distribution."""
        return weighted_quantile(self.value_dist(f_pi), self.weights, alpha)

    def posterior_mean(self):
        return self.weights @ self.samples

    def value_mean(self, f_pi):
        return float(self.weights @ self.value_dist(f_pi))

    def diagnostics(self):
        return dict(ess=self.ess, cov_trace=float(np.trace(self.cov)),
                    map_w=self.map_w.tolist())

    # ---- MH validation sampler (PG5) ----------------------------------
    def mh_sample(self, dPhi, y, betaT, prior_scale, n=20000, burn=2000,
                  step=None, rng=None):
        dPhi = np.asarray(dPhi, float); y = np.asarray(y, float)
        s2 = prior_scale ** 2
        rng = rng or np.random.default_rng(0)
        if step is None:
            step = np.sqrt(np.diag(self.cov))
        def logpost(w):
            z = betaT * (dPhi @ w)
            return -(np.sum(y * _softplus(-z) + (1 - y) * _softplus(z))
                     + 0.5 * (w @ w) / s2)
        w = self.map_w.copy(); lp = logpost(w)
        out = []
        acc = 0
        for t in range(n + burn):
            prop = w + step * rng.standard_normal(self.k)
            lpp = logpost(prop)
            if np.log(rng.random()) < lpp - lp:
                w, lp = prop, lpp; acc += 1
            if t >= burn:
                out.append(w.copy())
        self.mh_accept = acc / (n + burn)
        return np.array(out)
