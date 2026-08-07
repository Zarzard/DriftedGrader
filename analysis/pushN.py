"""Endogenous null-space push (Paper B go/no-go): does ordinary closed-loop
training against a GENERIC-blind reward model drive the satiating grader's weight
into the model's null space N?

Mechanism under test: the policy over-produces the S-features the blind RM
rewards; satiation drains weight from what is produced; simplex renormalization
redistributes the freed mass -- including into N. If true, ||Pi_N w_t|| grows
FASTER under the closed-blind loop than under controls that share the same arms,
grader dynamics, and w0 but deploy different policies.

Arms (paired per seed: identical arms instance, same fixed generic projection P
with null space N = span-complement; DriftSat is deterministic given m):
  closed_blind  policy = Gibbs(posterior-mean of the k=3 P-projected posterior)
                -- the treatment (run_p1 protocol verbatim)
  closed_full   policy = Gibbs(posterior-mean of the full k=6 posterior); Pi_N
                measured against the SAME N -- controls for "any competent
                closed loop" (well-specified tracking)
  oracle        policy = Gibbs(Phi w_t) -- perfect grader tracking, no learning
  random        policy = uniform every round -- no optimization at all

(A replay control is omitted BY CONSTRUCTION: replay copies the treatment's
w-trajectory verbatim, so its ||Pi_N w_t|| is identical -- it cannot
discriminate. The random/oracle/closed_full arms are the informative ones.)

STATISTIC (normalization-confound guard): the simplex fixes ||w||_1=1 but NOT
||w||_2 (range [1/sqrt(d), 1]); raw ||Pi_N w||_2 scales with ||w||_2, so it
conflates rotation INTO N with L2 concentration of w -- and closed loops change
concentration strongly. The verdict therefore gates on the scale-invariant
DIRECTION fraction  frac_t = ||Pi_N w_t|| / ||w_t||_2  (random-direction null:
sqrt((d-k)/d) = sqrt(3/6) ~ 0.707). Raw ||Pi_N w|| and ||w||_2 are reported as
the decomposition.

Verdict: PUSH CONFIRMED iff Delta-frac (last-10 mean minus t=0) for closed_blind
exceeds ALL controls with a seed-PAIRED bootstrap 95% CI excluding 0.
Also correlates per-seed push with the treatment's final rank_corr/regret
(does the push predict the damage?).

Run: python -m analysis.pushN --out <dir> --seeds 24 --workers 16
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from drifted_grader.arms import make_arms
from drifted_grader.graders import make_grader
from drifted_grader.reward_models import make_projection
from drifted_grader.metrics import policy_from_scores, rangenorm, regret_Rt, rank_correlation
from drifted_grader.posterior import RewardPosterior
from drifted_grader.policies import expected_features, sample_pairs

N, D, K, BETA, BETAT, Q, PRIOR = 256, 6, 3, 0.05, 10.0, 64, 2.0
NS, BUF_CAP, T, LAST = 4000, 1800, 100, 10
ARMS = ("closed_blind", "closed_full", "oracle", "random")


def run_arm(seed, lam, arm):
    """One trajectory. Arms/grader/projection are seed-paired; only the deployed
    policy differs. Returns per-round ||Pi_N w_t|| (+ diagnostics)."""
    ss = np.random.SeedSequence(seed)
    prng = np.random.default_rng(ss.spawn(1)[0])
    arng = np.random.default_rng(np.random.SeedSequence(seed + 987654).spawn(1)[0])
    arms = make_arms(N, D, prng)
    Phi, w0 = arms.Phi, arms.w0
    # the SAME generic-blind projection for every arm at this seed (measurement
    # target; also the learner's projection in closed_blind)
    P = make_projection(D, K, np.random.default_rng(seed + 555))
    PiN = np.eye(D) - P.T @ P                        # rows of P orthonormal
    grader = make_grader("SAT", arms, BETAT, prng, lam=lam, T=T)

    needs_fit = arm in ("closed_blind", "closed_full")
    Pfit = P if arm == "closed_blind" else None      # None -> full features
    post = RewardPosterior(K if arm == "closed_blind" else D)
    buf_d, buf_y = [], []
    m = np.zeros(D)
    pnw, wn, frac, pnm, wdist = [], [], [], [], []
    dmg = {}
    for t in range(T):
        w_t = grader.w.copy()
        nw = float(np.linalg.norm(w_t))
        pn = float(np.linalg.norm(PiN @ w_t))
        pnw.append(pn); wn.append(nw); frac.append(pn / (nw + 1e-12))
        wdist.append(float(np.linalg.norm(w_t - w0)))

        if arm == "random":
            pi = np.full(N, 1.0 / N)
        elif arm == "oracle":
            pi = policy_from_scores(rangenorm(Phi @ w_t), BETA)
        else:
            if buf_y:
                Z = Phi @ Pfit.T if Pfit is not None else Phi
                pi = policy_from_scores(rangenorm(Z @ post.posterior_mean()), BETA)
            else:
                pi = np.full(N, 1.0 / N)

        if needs_fit:
            pairs = sample_pairs(pi, Q, arng)
            y = grader.label(pairs, arng)
            buf_d.append(Phi[pairs[:, 0]] - Phi[pairs[:, 1]]); buf_y.append(y)
            alld = np.concatenate(buf_d); ally = np.concatenate(buf_y).astype(float)
            if len(ally) > BUF_CAP:
                idx = arng.choice(len(ally), BUF_CAP, replace=False)
                alld, ally = alld[idx], ally[idx]
            dz = alld @ Pfit.T if Pfit is not None else alld
            post.fit(dz, ally, BETAT, prior_scale=PRIOR, n_samples=NS, rng=arng)

        # damage metrics for the treatment at the FINAL round, against the w_t
        # the policy actually faced (i.e. BEFORE the last grader.step)
        if arm == "closed_blind" and t == T - 1 and buf_y:
            evalp = [policy_from_scores(Phi @ w0, b) for b in (2.0, 0.2, 0.05)]
            evalp.append(np.full(N, 1.0 / N))
            o = np.zeros(N); o[(Phi @ w0).argmax()] = 1.0; evalp.append(o)
            lb = [post.lower_bound(P @ expected_features(pe, Phi), 0.05) for pe in evalp]
            v = [float(w_t @ expected_features(pe, Phi)) for pe in evalp]
            dmg["rank_corr_T"] = rank_correlation(lb, v)
            dmg["R_T"], _ = regret_Rt(pi, Phi, w_t, BETA)

        f = expected_features(pi, Phi)
        pnm.append(float(np.linalg.norm(PiN @ f) / (np.linalg.norm(f) + 1e-12)))
        m = 0.5 * m + 0.5 * f
        grader.step(m)

    out = dict(seed=int(seed), lam=lam, arm=arm,
               frac=[round(v, 5) for v in frac],
               frac0=frac[0], frac_last=float(np.mean(frac[-LAST:])),
               dfrac=float(np.mean(frac[-LAST:]) - frac[0]),
               pnw0=pnw[0], pnw_last=float(np.mean(pnw[-LAST:])),
               dpnw=float(np.mean(pnw[-LAST:]) - pnw[0]),
               wn0=wn[0], wn_last=float(np.mean(wn[-LAST:])),
               dwn=float(np.mean(wn[-LAST:]) - wn[0]),
               pnm_mean=float(np.mean(pnm)),
               wdist_last=float(np.mean(wdist[-LAST:])))
    out.update(dmg)
    return out


def _worker(args):
    (lam, arm), seed = args
    try:
        return run_arm(seed, lam, arm), None
    except Exception as e:
        return None, f"lam={lam} arm={arm} seed={seed}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [((lam, arm), s) for lam in args.lams for arm in ARMS
            for s in range(args.seeds)]
    print(f"{len(jobs)} runs ({len(args.lams)} lams x {len(ARMS)} arms x "
          f"{args.seeds} seeds)", flush=True)

    rows, crashes = [], []
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        for r, err in pool.imap_unordered(_worker, jobs, chunksize=1):
            (rows.append(r) if r else crashes.append(err))
    for c in crashes:
        print("CRASH:", c)
    json.dump(rows, open(os.path.join(args.out, "pushN.json"), "w"), indent=1)

    rng = np.random.default_rng(0)
    def paired_ci(diffs, nb=4000):
        d = np.asarray(diffs, float)
        bs = np.array([np.mean(rng.choice(d, len(d))) for _ in range(nb)])
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    lines = ["# Endogenous null-space push (Paper B go/no-go)\n",
             f"N={N} d={D} k={K} T={T}; {args.seeds} seeds, paired across arms "
             "(same arms/w0/P per seed). VERDICT statistic = Delta of the "
             "scale-invariant direction fraction frac_t = ||Pi_N w_t||/||w_t||_2 "
             f"(last-{LAST}-round mean minus t=0); random-direction null "
             f"sqrt((d-k)/d) = {np.sqrt((D - K) / D):.3f}. Raw ||Pi_N w|| and "
             "||w||_2 are shown as the decomposition (the simplex fixes ||w||_1, "
             "not ||w||_2, so the raw statistic conflates rotation with "
             "concentration). PUSH CONFIRMED iff closed_blind minus EVERY control "
             "> 0 on Delta-frac with a seed-paired 95% bootstrap CI excluding 0.\n"]
    verdicts = []
    for lam in args.lams:
        lines.append(f"\n## lam={lam:g}\n")
        lines.append("| arm | frac_0 | frac_T | Delta-frac | ||Pi_N w||_T (raw) "
                     "| ||w||_2,T | Delta||w||_2 | Pi_N-frac of produced m |")
        lines.append("|---|---|---|---|---|---|---|---|")
        per = {}
        for arm in ARMS:
            sel = {r["seed"]: r for r in rows if r["lam"] == lam and r["arm"] == arm}
            per[arm] = sel
            v = list(sel.values())
            lines.append(f"| {arm} | {np.mean([r['frac0'] for r in v]):.3f} "
                         f"| {np.mean([r['frac_last'] for r in v]):.3f} "
                         f"| {np.mean([r['dfrac'] for r in v]):+.3f} "
                         f"| {np.mean([r['pnw_last'] for r in v]):.3f} "
                         f"| {np.mean([r['wn_last'] for r in v]):.3f} "
                         f"| {np.mean([r['dwn'] for r in v]):+.3f} "
                         f"| {np.mean([r['pnm_mean'] for r in v]):.3f} |")
        lines.append("\n**Seed-paired contrasts on Delta-frac (closed_blind - control):**\n")
        all_pos = True
        for ctrl in ("closed_full", "oracle", "random"):
            common = sorted(set(per["closed_blind"]) & set(per[ctrl]))
            diffs = [per["closed_blind"][s]["dfrac"] - per[ctrl][s]["dfrac"]
                     for s in common]
            mdiff, lo, hi = paired_ci(diffs)
            sig = "SIGNIFICANT" if lo > 0 else ("negative" if hi < 0 else "ns")
            all_pos &= lo > 0
            lines.append(f"- vs {ctrl}: {mdiff:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig} "
                         f"(n={len(common)} paired seeds)")
        verdicts.append((lam, all_pos))
        # does the push predict the damage?
        cb = [per["closed_blind"][s] for s in sorted(per["closed_blind"])]
        cb = [r for r in cb if "rank_corr_T" in r]
        if len(cb) >= 5:
            sp = rank_correlation([r["dfrac"] for r in cb], [r["rank_corr_T"] for r in cb])
            sr = rank_correlation([r["dfrac"] for r in cb], [r["R_T"] for r in cb])
            lines.append(f"- push->damage: Spearman(dfrac, rank_corr_T) = {sp:+.2f}; "
                         f"Spearman(dfrac, regret_T) = {sr:+.2f}  (n={len(cb)} seeds)")

    lines.append("\n## Verdict\n")
    for lam, ok in verdicts:
        lines.append(f"- lam={lam:g}: {'PUSH CONFIRMED' if ok else 'NOT confirmed'}")
    lines.append("\n(Replay control omitted by construction: it copies the "
                 "treatment's w-trajectory, so its ||Pi_N w_t|| is identical and "
                 "cannot discriminate.)\n")
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "results.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
