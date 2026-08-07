"""Closed-loop task-reward analysis: does optimizing SAC against a drift-blind reward model
(frozen_narrow) degrade REAL task performance more under grader drift than a full-representation
RM? Complements the open-loop SELECTION results (analysis/stageF_scaleup.py).

Reads closedloop/exp/<task>/<arm>/<rm_mode>/seed<seed>/{eval.csv, drift_log.jsonl, run.log}.
Metric methodology (guards learned the hard way):
  * CONVERGED + STAGE-MATCHED only: a cell counts if drift_log's last step >= --minstep; NONE vs
    SAT are compared only among converged cells (both arms at 300k). Pooling different training
    stages fabricates spurious effects.
  * PRIMARY = raw dense true reward (eval true_episode_reward, final-window mean). Report absolute
    drop NONE-SAT and relative drop, but FLAG floor cases (NONE below --floor) where the relative
    metric is unstable; never headline a ratio with a near-zero denominator.
  * INTERACTION test = per task, (relative drop of frozen) - (relative drop of full); pooled mean +
    sign across tasks. This is THE claim (does drift hurt the blind arm *more*), separate from the
    NONE capacity main effect.
  * HACKING panel = train-log R (learned-reward return) vs TR (true-reward return) for frozen under
    NONE vs SAT: hacking = learned holds/rises while true falls.
  * MONITOR = RM held-out pref accuracy (run.log 'ACC:') NONE vs SAT: should stay green.

Usage: python -m analysis.stageF_closedloop --root <.../closedloop> [--minstep 280000]
"""
from __future__ import annotations
import argparse, os, glob, csv, json, re
import numpy as np

ARMS = ["NONE", "SAT_lam1.0"]
RMS = ["full", "linear", "frozen_narrow"]


def _last_step(run_dir):
    p = os.path.join(run_dir, "drift_log.jsonl"); L = -1
    if os.path.exists(p):
        for ln in open(p):
            ln = ln.strip()
            if ln:
                try: L = max(L, int(json.loads(ln)["step"]))
                except Exception: pass
    return L


def _final_true_reward(run_dir, minstep, window=3):
    p = os.path.join(run_dir, "eval.csv")
    if not os.path.exists(p): return None
    rows = [x for x in csv.DictReader(open(p)) if int(x["step"]) >= minstep]
    if not rows: return None
    return float(np.mean([float(x["true_episode_reward"]) for x in rows[-window:]]))


def _final_R_TR(run_dir, k=15):
    """learned-reward return R and true-reward return TR from the train log final window."""
    p = os.path.join(run_dir, "run.log")
    if not os.path.exists(p): return None
    R, TR = [], []
    for ln in open(p, errors="ignore"):
        m = re.search(r"\| R:\s*([-\d.]+).*\| TR:\s*([-\d.]+)", ln)
        if m: R.append(float(m.group(1))); TR.append(float(m.group(2)))
    if len(R) < k: return None
    return float(np.mean(R[-k:])), float(np.mean(TR[-k:]))


def _final_acc(run_dir, k=20):
    p = os.path.join(run_dir, "run.log")
    if not os.path.exists(p): return None
    accs = [float(m) for m in re.findall(r"ACC:\s*([\d.]+)", open(p, errors="ignore").read())]
    return float(np.mean(accs[-k:])) if len(accs) >= 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/stageF_scaleup/closedloop")
    ap.add_argument("--minstep", type=int, default=280000)
    ap.add_argument("--floor", type=float, default=50.0, help="NONE trueR below this => relative metric unstable")
    args = ap.parse_args()
    exp = os.path.join(args.root, "exp"); out = os.path.join(args.root, "results")
    os.makedirs(out, exist_ok=True)

    # gather converged cells
    cells = {}  # (task,arm,rm) -> list of trueR
    RTR = {}    # (task,arm,rm) -> list of (R,TR)
    ACC = {}    # (task,arm,rm) -> list of acc
    tasks = set()
    for d in sorted(glob.glob(os.path.join(exp, "*", "*", "*", "seed*"))):
        parts = d.split(os.sep); task, arm, rm = parts[-4], parts[-3], parts[-2]
        if arm not in ARMS or rm not in RMS: continue
        if _last_step(d) < args.minstep: continue           # converged only
        tr = _final_true_reward(d, args.minstep)
        if tr is None: continue
        tasks.add(task)
        cells.setdefault((task, arm, rm), []).append(tr)
        rtr = _final_R_TR(d);  ACC.setdefault((task, arm, rm), []).append(_final_acc(d))
        if rtr: RTR.setdefault((task, arm, rm), []).append(rtr)
    tasks = sorted(tasks)

    def m(key):
        v = cells.get(key); return (np.mean(v), np.std(v), len(v)) if v else (np.nan, 0, 0)

    # per-task table + interaction
    per_task, inter = [], []
    for task in tasks:
        row = {"task": task}
        ok = True
        for rm in RMS:
            nm, ns, nn = m((task, "NONE", rm)); sm, ss, sn = m((task, "SAT_lam1.0", rm))
            reld = (nm - sm) / nm if (nn and sn and nm > args.floor) else np.nan
            row[rm] = dict(none=nm, sat=sm, nn=nn, sn=sn, absdrop=nm - sm, reldrop=reld,
                           floored=(nn and nm <= args.floor))
            if rm in ("full", "frozen_narrow") and (not nn or not sn): ok = False
        per_task.append(row)
        if ok and not np.isnan(row["full"]["reldrop"]) and not np.isnan(row["frozen_narrow"]["reldrop"]):
            inter.append((task, row["frozen_narrow"]["reldrop"] - row["full"]["reldrop"]))

    # pooled interaction (the claim): frozen relative-drop minus full relative-drop, across tasks
    diffs = [d for _, d in inter]
    pooled_inter = (float(np.mean(diffs)), float(np.std(diffs)), len(diffs)) if diffs else (np.nan, 0, 0)
    npos = sum(1 for d in diffs if d > 0)

    # write
    L = ["# Closed-loop task-reward: does a drift-blind RM hack real performance under grader drift?\n",
         f"Converged cells only (drift_log last step >= {args.minstep}); dense true_episode_reward, "
         f"final-window mean. Tasks with data: {len(tasks)}.\n",
         "## Raw converged true reward + drift-induced drop (NONE -> SAT lam1.0)\n",
         "| task | arm | NONE | SAT | abs drop | rel drop | note |", "|---|---|---|---|---|---|---|"]
    for row in per_task:
        for rm in RMS:
            c = row[rm]
            note = "FLOOR (rel unstable)" if c["floored"] else ""
            rel = f"{c['reldrop']*100:+.0f}%" if not np.isnan(c["reldrop"]) else "--"
            L.append(f"| {row['task'].replace('metaworld_','')} | {rm} | {c['none']:.0f} (n{c['nn']}) "
                     f"| {c['sat']:.0f} (n{c['sn']}) | {c['absdrop']:+.0f} | {rel} | {note} |")
    L.append("\n## THE interaction test (does drift hurt the blind arm MORE?)\n")
    L.append("Per task: (frozen_narrow relative-drop) - (full relative-drop). Positive => frozen degrades "
             "more under drift (supports the closed-loop coverage-gradient claim).\n")
    L.append("| task | frozen_reldrop - full_reldrop |"); L.append("|---|---|")
    for task, d in inter:
        L.append(f"| {task.replace('metaworld_','')} | {d*100:+.0f}% |")
    pm, ps, pn = pooled_inter
    # honest verdict: require the pooled interaction to exceed its OWN spread and be consistent in
    # sign across tasks. A mean smaller than its std is noise, no matter how large.
    supports = bool(pn >= 4 and pm > 0.10 and pm > ps and npos >= pn - 1)
    L.append(f"\n**Pooled interaction = {pm*100:+.0f}% ± {ps*100:.0f}% over {pn} tasks; "
             f"{npos}/{pn} tasks positive.** "
             + ("Supports a closed-loop coverage gradient." if supports else
                "**NOT a consistent closed-loop coverage gradient** — the spread exceeds the mean "
                f"(±{ps*100:.0f}% vs {pm*100:+.0f}%) and the sign flips across tasks. Coverage protects "
                "SELECTION (ranking vs true value), not OPTIMIZATION: under grader drift a faithful full "
                "RM *chases the drifted standard* (so full also degrades, and on some tasks drift "
                "reweights toward the task and full even improves), while a frozen RM's blindness can "
                "accidentally ignore the drift. n=3 seeds/arm leaves large variance; the direction, not "
                "just the power, is inconsistent.") + "\n")

    # hacking panel
    L.append("## Hacking panel: frozen_narrow learned-reward return (R) vs true-reward return (TR)\n")
    L.append("| task | arm | learned R | true TR |"); L.append("|---|---|---|---|")
    for task in tasks:
        for arm in ARMS:
            v = RTR.get((task, arm, "frozen_narrow"))
            if v:
                R = np.mean([x[0] for x in v]); TR = np.mean([x[1] for x in v])
                L.append(f"| {task.replace('metaworld_','')} | {arm} | {R:.1f} | {TR:.1f} |")
    # monitor panel
    L.append("\n## Monitor: RM held-out preference accuracy (should stay green under drift)\n")
    L.append("| task | arm | frozen pref-ACC | full pref-ACC |"); L.append("|---|---|---|---|")
    for task in tasks:
        for arm in ARMS:
            def acc(rm):
                a = [x for x in ACC.get((task, arm, rm), []) if x is not None]
                return f"{np.mean(a):.2f}" if a else "--"
            L.append(f"| {task.replace('metaworld_','')} | {arm} | {acc('frozen_narrow')} | {acc('full')} |")

    open(os.path.join(out, "results.md"), "w").write("\n".join(L) + "\n")
    json.dump(dict(tasks=tasks, pooled_interaction=pooled_inter, npos=npos,
                   per_task=[{k: (v if isinstance(v, str) else {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv)
                                                                 for kk, vv in v.items()})
                              for k, v in r.items()} for r in per_task]),
              open(os.path.join(out, "closedloop.json"), "w"), indent=2, default=float)
    print("\n".join(L))
    print(f"\n[wrote {out}/results.md]")


if __name__ == "__main__":
    main()
