# CHPC Resources Guide — DriftedGrader / NeuralDriftedGrader

*A handover for the next coding agent: how to actually run this project on the University of Utah CHPC cluster — machines, storage, conda envs, SLURM, and the job-orchestration pattern. Grounded in the real scripts under `NeuralDriftedGrader/stageF_scaleup/`. For the **science** (what to run and why) see [STAGE_F_HANDOVER.md](STAGE_F_HANDOVER.md) and [PAPER.md](PAPER.md); note that doc's `/home/zifan/...` paths are stale — the real box uses the `/uufs/.../u1520755/...` paths below.*

---

## 0. Golden rules (read first)

1. **Never run heavy compute on the login node.** The notchpeak login node has only a tiny GT 1030 GPU. All training / LLM-judge / MuJoCo work goes through `sbatch`. Light things are fine on login: editing, `git`, reading CSV/JSON, plotting with matplotlib, `squeue`/`sacct`, tiny numpy.
2. **Write big outputs to the group space, not `$HOME`.** Home has a small quota. Everything large (runs, checkpoints, logs, HF cache) goes under `…/dbrown-group1/zifan/` (or `/scratch`). See §2.
3. **Match `--account` to `--partition`.** CHPC rejects mismatches. The valid pairs are in §3.
4. **Granite is a separate cluster.** Jobs to it need `#SBATCH --clusters=granite`, and you monitor/cancel them with `squeue -M granite` / `scancel -M granite`.
5. **Every MuJoCo job needs the env boilerplate** (`MUJOCO_GL=egl` + `LD_LIBRARY_PATH` + `module load cuda`). See §4. Missing it is the #1 cause of import crashes.

---

## 1. The two repos

| Repo | Path on this box | Purpose |
|---|---|---|
| **NeuralDriftedGrader** | `/uufs/chpc.utah.edu/common/home/u1520755/NeuralDriftedGrader` | PEBBLE + Stage-F drift code; produces the training runs. GitHub `Zarzard/NeuralDriftedGrader`, branch `main` (memory notes an older `perfg_rew_hack` branch — verify with `git branch`). |
| **DriftedGrader** | `/uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader` | Tabular simulator + all analysis/figures (`analysis/`, `drifted_grader/`), the paper (`md_files/`). GitHub `Zarzard/DriftedGrader`, branch `main`. |

All the SLURM orchestration lives in **`NeuralDriftedGrader/stageF_scaleup/`**.

---

## 2. Storage map

Check quotas anytime with **`mychpc storage`**. Current state (2026-08-05):

| Location | Quota / used | Use it for | Notes |
|---|---|---|---|
| `…/home/u1520755/` (`$HOME`) | **49.4 GiB, ~76 % used** | the two git repos, small source files | **Getting tight** — **do not** put runs/checkpoints here. |
| `…/home/dbrown-group1/zifan/` | **3 TiB group quota, ~86 % full** | **all big outputs, conda envs, results** | Group-owned. `miniconda3/`, `stageF_scaleup/` (~87 GB of runs), `llm_judge_drift/`. The "scratch or dbrown-group1" the project standardized on. **Near full — clean up stale runs before big new batches** (checkpoints dominate; a 300k run dir is ~hundreds of MB). |
| `/scratch/general/vast/u1520755/` | huge (PB-scale shared), ~7 GiB used | HuggingFace cache, transient fast I/O | Fast parallel FS. `HF_HOME` points here for LLM jobs. Scratch is **auto-purged** (~60 days idle) — never the only copy of anything. |
| `/scratch/general/{nfs1,lustre}/` | huge, shared | alternative scratch | Same purge policy. Good spill target if `dbrown-group1` fills. |

> **If the group space is full and a batch fails on write**, either prune completed runs' checkpoints (`*.pt`) — the analyzers only need `drift_log.jsonl` + `ranking_export.pkl` + `eval.csv` — or redirect new `run_dir`s to `/scratch/general/vast/u1520755/…` in `gen_cells.py`.

Canonical output tree (already populated):
```
…/dbrown-group1/zifan/
├── miniconda3/                 # all conda envs live here
├── stageF_scaleup/
│   ├── exp/<task>/<arm>/<seed>/    # PEBBLE run dirs (drift_log.jsonl, ranking_export.pkl, *.pt, eval.csv)
│   ├── logs/                       # SLURM stdout (%x_%A_%a.out)
│   ├── manifests/                  # stageF_cells.json  <- the array manifest
│   ├── results*/per_run.csv        # analyzer outputs (results/, results_repr/, results_hack/, …)
│   └── closedloop/ morl/ …         # other experiment families
└── llm_judge_drift/
    ├── coverage_law/law.json  semantic/sem.json  refine/refine.json  …
    └── logs/
```

---

## 3. Compute: clusters, partitions, accounts

You have access to owner (dbrown) hardware plus general/guest allocations. **Use in this priority order:**

| Partition | Account | (QOS / cluster) | Hardware | When to use |
|---|---|---|---|---|
| `dbrown-gpu-np` | `dbrown-gpu-np` | notchpeak | **notch367 = 8× A6000**, non-preemptable | **Default workhorse.** Reliable, owner priority. |
| `soc-gpu-np` | `soc-gpu-np` | notchpeak, **preemptable** | shared A-series | **Spill** for extra parallelism. Always add `--requeue`. |
| `dbrown-gpu-grn` | `dbrown` (QOS `dbrown-gpu-grn`) | **granite** (`--clusters=granite`) | **grn015 = 4 GPU, 76 CPU, 320 GB**, non-preemptable | One **fat** allocation running many cells concurrently (§5, granite pool). |
| `rai-gpu-grn` | `rai` (QOS `rai-gpu-grn`) | granite | 1 GPU | LLM/vision-judge jobs (used by `llmjudge.slurm`). |
| `notchpeak-gpu` | your general GPU alloc | notchpeak | shared | Fallback when owner nodes are full. |
| `notchpeak-shared-short` | (general) | notchpeak, ≤8 h | shared | Quick **interactive debug** / smoke tests. |

Other visible partitions (`owner-gpu-guest`, `kingspeak-gpu`, `lonepeak-gpu`, `soc-gpu-kp`…) are guest/preemptable fallbacks — free but can be killed; use `--requeue`.

Check live state:
```bash
sinfo -p dbrown-gpu-np -o "%P %a %D %t"          # owner node availability
squeue -u $USER                                   # your notchpeak jobs
squeue -u $USER -M granite                         # your granite jobs (separate cluster!)
sacctmgr -n show assoc user=$USER format=Account,Partition   # what you may use
```

---

## 4. Per-job environment boilerplate

Every `.slurm` here sets these. If you write a new job, copy them.

**MuJoCo / PEBBLE runs (`bpref` or `bprefmo` env):**
```bash
module load cuda/12.4.0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4           # avoid BLAS oversubscription
export MUJOCO_GL=egl                                  # headless GL — REQUIRED
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
```

**LLM / vision judge (`postdex` env):**
```bash
export HF_HOME=/scratch/general/vast/u1520755/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```
(Models are pre-cached; offline flags avoid a network stall on compute nodes.)

**Hydra gotchas (old Hydra in NeuralDriftedGrader):**
- It does **not** resolve `${oc.env:...}` in the output-dir template → **always pass `hydra.run.dir=<abs path>` explicitly.**
- Config is `strict=True` → only pass keys that already exist.
- `env=metaworld_<task>-v2` is a plain string param, not a Hydra config group.

---

## 5. Conda environments

Activate with the shared miniconda, or just call the env's python by absolute path (what the SLURM scripts do — no `conda activate` needed).

```bash
source /uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/etc/profile.d/conda.sh
conda activate bpref
# …or directly:
PY=/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/bpref/bin/python
```

| Env | Has | Use for |
|---|---|---|
| **`bpref`** | mujoco_py, torch(+cuda), pandas, matplotlib (py3.8) | **PEBBLE Metaworld drift runs; all `analysis/` scripts and figure generation.** |
| **`bprefmo`** | + gymnasium, mo_gymnasium | MO-Gymnasium closed-loop experiments. |
| **`postdex`** | torch, transformers, gymnasium | LLM-judge (Qwen2.5-3B) and vision-judge jobs. |
| **`morl`** | gymnasium, mo_gymnasium | MORL baselines. |
| `jaxrl`, `multif` | — | other/legacy; unused by the paper. |

> Figures for the paper regenerate under `bpref`: `conda activate bpref && python -m analysis.paper_figures` (reads stored CSV/JSON, writes `md_files/figures/`). This is light — fine on the login node.

---

## 6. The orchestration pattern

Everything scales through **one manifest + an idempotent array (or pool) job**. Three moving parts:

1. **`gen_cells.py`** → writes `manifests/stageF_cells.json`: a list of *cells*, each `{run_name, run_dir, num_train_steps, command}`. The `command` is a full `train_PEBBLE.py …` invocation with an absolute `hydra.run.dir`. Variants exist per experiment family: `gen_cells_cl.py` (closed-loop), `gen_cells_mo.py` / `gen_cells_modir.py` (MO-Gym).
2. **A runner** reads a cell and executes it, with **skip-if-done**: a run counts as complete iff `drift_log.jsonl`'s last `step` ≥ 95 % of `num_train_steps`. Re-submitting is therefore safe and resumes only what's missing. Two runner styles:
   - **`stageF_array.slurm`** — SLURM **array**, one array task = one cell (`SLURM_ARRAY_TASK_ID` indexes the manifest). One GPU per task. Concurrency capped with `%N`.
   - **`granite_pool.slurm`** — **one fat allocation**, a bash job-pool running `STAGEF_CONC` cells concurrently across the node's 4 GPUs (reads an index list from `$STAGEF_IDXFILE`). Because the bottleneck is CPU env-stepping, it **oversubscribes GPUs** (multiple runs per GPU).
3. **`submit.sh`** — regenerates the manifest, then splits the array **~60 % to the owner node** (`dbrown-gpu-np`) and **~40 % spill** to `soc-gpu-np --requeue`. Idempotent.

Throughput reference: state-based Metaworld ≈ 100–160 env-steps/s → a 300k-step run ≈ 30–60 min. **GPU is not the bottleneck** (tiny MLPs); CPU env-stepping is — hence oversubscription. ~12–16 concurrent runs fit a 24-core node.

---

## 7. Copy-paste recipes

**a) Smoke test (validate the code path before a big batch, ~15 min):**
```bash
cd /uufs/chpc.utah.edu/common/home/u1520755/NeuralDriftedGrader/stageF_scaleup
sbatch smoke.slurm            # short SAT run; checks drift_log.jsonl + ranking_export.pkl load
```

**b) Submit the full drift batch (array, owner + soc spill):**
```bash
cd …/NeuralDriftedGrader/stageF_scaleup
DRY_RUN=true bash submit.sh     # preview the array split, submit nothing
bash submit.sh                  # generate manifest + submit
OWNER_COUNT=120 bash submit.sh  # force everything onto the owner node (no spill)
```

**c) Booster: one fat granite allocation over a chosen index list:**
```bash
printf '%s\n' 0 1 2 3 4 5 > /tmp/idx.txt
STAGEF_IDXFILE=/tmp/idx.txt STAGEF_CONC=18 sbatch granite_pool.slurm
```

**d) LLM-judge GPU job (postdex env, granite `rai` partition):**
```bash
sbatch llmjudge.slurm           # runs analysis.llm_judge_demo, writes llm_judge_drift/results/
# siblings: covlaw.slurm, semantic.slurm, llmjudge_refine.slurm, visionjudge.slurm, nlprobe.slurm
```

**e) Analyze finished runs (produces the per_run.csv the paper/figures read):**
```bash
sbatch analyze.slurm                       # or, for light analyzers, on login under bpref:
conda activate bpref
python -m analysis.stageF_scaleup --exp_glob '…/dbrown-group1/zifan/stageF_scaleup/exp/*' --with_mlp
```

**f) Regenerate paper figures (light, login-node OK):**
```bash
conda activate bpref
cd /uufs/chpc.utah.edu/common/home/u1520755/DriftedGrader
python -m analysis.paper_figures            # -> md_files/figures/fig{1..5}_*.png
```

---

## 8. Monitoring & etiquette

```bash
squeue -u $USER                         # notchpeak queue
squeue -u $USER -M granite              # granite queue (separate cluster)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,NodeList   # post-mortem
tail -f …/dbrown-group1/zifan/stageF_scaleup/logs/<job>_<A>_<a>.out   # live log
scancel <jobid>            /   scancel -M granite <jobid>        # cancel
scancel -u $USER --state=PENDING                                 # drop your queued jobs
mychpc storage                          # home / scratch / group-space quota + % full
```
- SLURM stdout goes to `…/stageF_scaleup/logs/%x_%A_%a.out`; each run also tees its own `run.log` into its `run_dir`.
- Prefer the owner node for correctness-critical runs (no preemption). Put throughput-only bulk on `soc`/guest with `--requeue`.
- Don't sit on the owner node with idle interactive allocations — others in `dbrown` share it.

---

## 9. Common failures → fixes

| Symptom | Cause | Fix |
|---|---|---|
| `mujoco_py` import / GL error at start | missing MuJoCo env | set `MUJOCO_GL=egl` **and** the `LD_LIBRARY_PATH` line (§4); `module load cuda/12.4.0`. |
| Output dir is literal `${oc.env:…}` | old Hydra can't resolve env interpolation | pass an **absolute** `hydra.run.dir=…`. |
| `sbatch: error: invalid account/partition` | account≠partition | use a valid pair from §3 (e.g. `--partition=dbrown-gpu-np --account=dbrown-gpu-np`). |
| Granite job "not found" in `squeue` | wrong cluster | it's on granite: `squeue -M granite`; submit needs `--clusters=granite`. |
| Job killed mid-run on `soc`/guest | preemption | expected; submit with `--requeue`; skip-if-done resumes it. |
| HF model download hangs on compute node | no/slow network | pre-cache to `HF_HOME=/scratch/general/vast/u1520755/huggingface`, set `HF_HUB_OFFLINE=1`. |
| `success_rate=0` under random-feature drift | synthetic grader is decoupled from task reward | **expected**, not a bug — see STAGE_F_HANDOVER §7. Use `drift_feature=selected` if you need task competence. |

---

## 10. Where to look next

- **Science / what-to-run:** [STAGE_F_HANDOVER.md](STAGE_F_HANDOVER.md) (§2 reproduction gate, §6 scale-up plan, §7 caveats) — remember its paths are stale; use the `/uufs/…/u1520755/…` paths here.
- **Results & narrative:** [PAPER.md](PAPER.md); numbers/log in `EXPERIMENT_LOG.md`; per-experiment `results.md` under `…/llm_judge_drift/*/`.
- **Analyzers:** `DriftedGrader/analysis/` (`stageF_scaleup.py`, `stageF_ranking.py`, `coverage_law.py`, `paper_figures.py`, …).
- **Job scripts:** `NeuralDriftedGrader/stageF_scaleup/*.slurm` + `submit*.sh` + `gen_cells*.py`.
- **CHPC docs:** https://www.chpc.utah.edu/documentation/ (SLURM, partitions, scratch policy).
