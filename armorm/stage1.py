"""Stage 1: the grader is an evaluator, never the thing the policy optimises.

Stage 0 let the policy maximise the grader's own score directly, which is not the arrangement the
paper is about. Here the chain is the one an RLHF pipeline actually uses:

    grader  ->  preference labels  ->  reward model (fitted)  ->  policy (LoRA)

The grader never touches the policy. It labels pairs. A reward model learns from those labels, and
the policy is trained against the reward model. Grader drift therefore has to survive a fitting step
to reach the policy at all, and a separate measurement put that transmission near 50% for a linear
fit, so Stage 0 overstates what arrives.

The loop also lets drift COMPOUND, which best-of-n over a fixed pool cannot: each round the policy
changes, so the next round's candidates are drawn from a different distribution. That is the part of
the motivating story Stage 0 structurally could not test.

Harm is measured against MATH's exact answer key on prompts the loop never trains on. The gold is
exact here, deliberately: the XSTest refusal label proved unreliable and would make every number an
interval.

Usage:
  python stage1.py --arm stationary --rounds 4 --out DIR
  python stage1.py --arm release_swap --drift-round 2 --rounds 4 --out DIR
  python stage1.py --arm matched_random --drift-round 2 --rounds 4 --out DIR --smoke
"""
import argparse
import gc
import json
import os
import time

from math import comb

import numpy as np
import torch

T0 = time.time()
def tick(m): print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)

POLICY = "Qwen/Qwen2.5-7B-Instruct"   # overridden by --policy
# bf16 generation is NOT reproducible across GPU architectures: two identical arms gave held-out
# 0.208 and 0.250 at round 0, before any training, on A6000 against H200. fp32 scoring WAS identical
# across hardware in 100% of pools, so an fp32 run removes the hardware dependence at roughly double
# the generation cost and double the memory. DTYPE is set once and used by every model load, so a
# run is internally consistent whichever is chosen.
DTYPE = torch.bfloat16
MATH_INSTR = ("Solve the problem. Reason briefly, then give the final answer in \\boxed{}.")

# Task-specific pieces, filled in by main(). A mutable dict rather than reassigned module globals:
# an earlier edit to this file raised "name 'POLICY' is used prior to global declaration", and
# mutating a dict needs no global statement anywhere.
TASK = dict(key="math", instr=MATH_INSTR, pred=None, same=None)
# the movement QRM v1 -> v2 made, measured on RewardBench in emb_frame.py and reused as the
# exogenous drift so the dose is a real release rather than a chosen number
MEASURED_DEG = 36.4


def free():
    gc.collect(); torch.cuda.empty_cache()


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def rotate(w, direction, deg):
    """Rotate each row of w toward `direction` by `deg`, keeping its norm."""
    d = direction - (direction * unit(w)).sum(1, keepdims=True) * unit(w)
    d = unit(d)
    th = np.radians(deg)
    return np.cos(th) * w + np.sin(th) * np.linalg.norm(w, axis=1, keepdims=True) * d


# Endogenous mechanisms belong HERE rather than in Stage 0. They are documented for human graders,
# and a human grader's role is to label, not to rerank at serving time; Stage 0b ran them in a
# selection loop where the picks trained nothing, so the drift could never reach a policy. In Stage 1
# the grader labels, a reward model learns the labels and a policy trains on them, which is both the
# role humans occupy and the only configuration where an adapting grader can compound.
ETA = 0.25            # how far the grader moves toward its own recent diet each round


# Per-process state for the mechanisms that need memory across rounds. One arm runs per process,
# so a plain module dict is safe and avoids threading extra arguments through _run_rounds.
_DRIFT_STATE = {}
ALPHA = 0.3           # acclimation memory: how fast the running average of delivered value moves
LAM = 1.0             # acclimation strength: how hard a sustained criterion is faded


def drift_angle(full, rnd, drift_round, schedule, seed=0, ramp=8, period=4):
    """How much of the total rotation has been applied by this round.

    Every exogenous arm was previously one-shot, which fixes drift SPEED at "instant" and leaves
    the timescale untested. That matters because visibility is a timescale rather than a property
    of the mechanism: the grader's own stream reports regret against the CURRENT weights, so a
    policy that can track a slow drift shows near-zero regret however far the weights have gone,
    while the same total movement delivered at once is visible until the policy catches up.
    Comparing `ramp` against `oneshot` at the SAME total angle is the test.
    """
    t = rnd - drift_round + 1
    if t <= 0:
        return 0.0
    if schedule == "oneshot":
        return full
    if schedule == "ramp":
        return full * min(1.0, t / float(ramp))
    if schedule == "oscillate":
        return full * float(np.sin(2 * np.pi * t / period))
    if schedule == "walk":
        g = np.random.default_rng(seed + 1000)
        steps = g.standard_normal(t)
        return full * float(np.sum(steps)) / np.sqrt(max(ramp, 1))
    raise ValueError(schedule)


def drift_w(w0, arm, rnd, drift_round, seed=0, profile=None, deg=None, schedule="oneshot",
            eta=None, lam=None, alpha=None, cumulative=False):
    """The grader's weights at this round.

    Exogenous arms are one-shot, matching documented pipeline maintenance. Endogenous arms move a
    little every round toward (or away from) the average criterion profile of what the grader has
    been selecting, which is what `profile` carries.

    eta/lam/alpha are the ENDOGENOUS intensity knobs, defaulting to the module constants so every
    arm run before they existed is reproduced exactly. They are the endogenous counterpart of
    --drift-deg: without them the endogenous arms cannot be intensity-matched to each other or to
    the rotations, and the four of them ran at 0.9-19.4 deg per round with no way to say so."""
    if cumulative and arm in ("acclimation", "habituation"):
        # Acclimation already carries history, through the r EWMA in _DRIFT_STATE. Applying the
        # cumulative blend to it as well would compound the same memory twice. Refuse rather than
        # silently ignore the flag.
        raise ValueError("--drift-cumulative is meaningless for acclimation: it already accumulates "
                         "through its r EWMA. It applies to satiation, self_preference, momentum.")
    eta = ETA if eta is None else float(eta)
    lam = LAM if lam is None else float(lam)
    alpha = ALPHA if alpha is None else float(alpha)
    if arm in ("stationary", "oracle", "release_swap_real"):
        # oracle is a positive control: the grader never moves, only the selection rule changes.
        # release_swap_real gets its drift by swapping to a different released model, so rotating
        # its weights on top of that would confound two channels.
        return w0
    if arm in ("release_swap", "matched_random"):
        if rnd < drift_round:
            return w0
        g = np.random.default_rng(seed)
        # The magnitude must be settable. MEASURED_DEG=36.4 came from an ArmoRM release pair and is
        # not the right scale for another grader: QRM v1->v2 has no equivalent angle at all, because
        # the criterion count changes. On UltraFeedback a rotation is instead calibrated to match the
        # real update's SELECTION-FLIP RATE (27.0%), which lands near 45 degrees -- and the point of
        # that arm is to document that a random direction of matched churn costs ~100x more gold
        # (-0.029, 11% of spread) than the real released update does (-0.0002, 0.1%).
        full = MEASURED_DEG if deg is None else deg
        th = drift_angle(full, rnd, drift_round, schedule, seed=seed)
        return rotate(w0, g.standard_normal(w0.shape), th)
    if profile is None:
        return w0
    z = profile / (np.linalg.norm(profile) + 1e-12)
    if arm == "satiation":              # attend to whatever the diet has been short of
        d = -z
    elif arm == "self_preference":      # attend to whatever the diet already delivers
        d = z
    elif arm in ("acclimation", "habituation"):
        # Helson's adaptation level in Lant's exponentially weighted moving-average form. Weights
        # decay multiplicatively with a RUNNING AVERAGE of delivered value, clamped at zero so only
        # criteria the policy is genuinely good at fade. `profile` is z-scored against the pool, so
        # it is signed, and the clamp therefore bites: without it, exp(-LAM*r) with r < 0 would
        # AMPLIFY the criteria the policy is worst at, which is satiation's target, not fade them.
        #
        # INTENSITY. This arm is the only endogenous one whose step size is NOT normalised. The
        # others blend a unit direction at a fixed ETA and so take a guaranteed 14-19 deg step every
        # round; this one's step is proportional to the spread of r across criteria, and r is an
        # ALPHA-EWMA of `profile`, which is itself already a 0.5-EWMA of the per-round z-profile.
        # Doubly smoothed, r is small and nearly uniform, the final renormalisation removes the
        # uniform part, and the measured step lands at 0.9-2.1 deg against satiation's 19.0. The
        # 0.002 flip rate follows from that, not from the mechanism. LAM is the knob; raising it
        # scales the angle roughly linearly while the angle is small.
        #
        # NAMING. This is the mechanism the paper and DRIFT_TYPES_LITERATURE call ACCLIMATION, and
        # the one tab_coverage.tex cites Helson 1964 and Lant 1992 for. It was briefly called
        # `habituation`, an invented label, while `acclimation` named a rotation approximation
        # (-|z| * sign(wbar)) that was cut for landing 10 degrees from satiation on these pools.
        # `habituation` survives ONLY as an alias so job 15892803_0, launched under that name,
        # can resume. Do not use it in new runs. Its output dir is uf12_habituation; the stale
        # uf*_acclimation dirs hold the CUT rotation arm, not this one.
        r = _DRIFT_STATE.get("r")
        r = profile.copy() * alpha if r is None else (1 - alpha) * r + alpha * profile
        _DRIFT_STATE["r"] = r
        w = w0 * np.exp(-lam * np.maximum(r, 0.0))
        return unit(w) * np.linalg.norm(w0, axis=1, keepdims=True)
    elif arm == "momentum":
        # Chase the DIRECTION the achieved profile most recently moved, not its level. This is the
        # one function of the profile the other endogenous arms do not cover: assimilation and
        # satiation key on where the policy IS, acclimation on where it has been, momentum on where
        # it is going. Round one has no previous profile, so it holds still.
        prev = _DRIFT_STATE.get("prev")
        _DRIFT_STATE["prev"] = profile.copy()
        if prev is None:
            return w0
        dm = profile - prev
        if np.linalg.norm(dm) < 1e-9:
            return w0
        d = dm / np.linalg.norm(dm)
    else:
        raise ValueError(arm)
    # RESET (default) vs CUMULATIVE. By default the blend starts from the CURRENT undrifted w0, so
    # the drifted grader sits a FIXED angle off-true every round and never wanders: satiation at
    # eta=0.25 measures 19.0 deg at round 2 and 19.0 deg at round 7. That models a grader whose
    # recalibration carries a systematic tilt of constant size.
    #
    # NEITHER FORM IS DOCUMENTED, and we should not imply one is. Our own citation audit found that
    # Helson 1964 states adaptation level as a weighted pooling of PRESENT stimuli, not a temporal
    # recursion; Lant 1992 fits free coefficients and does not give the EWMA closed form attributed
    # to it; Panickssery 2024 and Zheng 2023 measure a STATIC bias with no loop at all; and satiation
    # and momentum have no prior source. So whether an endogenous standard accumulates or resets is
    # entirely our modelling choice, for all four mechanisms. The choice is not free, though: the
    # reset is what keeps satiation in the no-harm regime, since cumulatively the same eta reaches
    # 99 deg by round 8 against 19 flat. That is why the cumulative arm is worth running -- as a
    # sensitivity check on an unsupported choice, not as a fidelity fix.
    #
    # Cumulative accumulates the drifted DIRECTION and keeps the magnitude from the current w0
    # (magnitude is null in an argmax). The known cost is that the drifted arm's direction then stops
    # inheriting the undrifted grader's own per-round recalibration, so this conflates "the standard
    # moved" with "the grader was not re-fit". Documented rather than hidden; the reset arm at the
    # same eta is the control that bounds it.
    base = unit(w0)
    if cumulative:
        prev = _DRIFT_STATE.get("cum")
        if prev is not None:
            base = prev
    step = unit((1 - eta) * base + eta * unit(np.broadcast_to(d, w0.shape)))
    if cumulative:
        _DRIFT_STATE["cum"] = step
    return step * np.linalg.norm(w0, axis=1, keepdims=True)


VLLM_PY = ("/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan/miniconda3/envs/"
           "vllmgen/bin/python")


def compose(q, fewshot=""):
    """The single definition of a user turn. Both backends call this, so they cannot diverge."""
    return f"{fewshot}{q}\n\n{TASK['instr']}"


def gen_samples_vllm(adapter, texts, n, temperature, max_new, seed=0):
    """Generate through vLLM in its own interpreter, handing data over as JSON on disk.

    Measured 0.693 s/prompt for 8 samples at max_new=400 on an A6000 against 4.83 s/prompt for
    HuggingFace generate, a 7.0x speedup (2456 vs ~250 output tok/s). A separate process is
    mandatory, not a convenience: vllmgen is python 3.11 with torch 2.13.0+cu130 while this process
    is python 3.8, so they can never share an interpreter.

    The environment has to be scrubbed. PYTHONPATH points at pylibs, built for python 3.8, which
    shadows vllmgen's 3.11 packages and fails with "No module named pydantic_core._pydantic_core";
    LD_LIBRARY_PATH points at the 3.8 env's libs. vllmgen/bin must be ON PATH because flashinfer
    shells out to `ninja` at first use, and every compile cache defaults under $HOME, which is small
    on this cluster.
    """
    import subprocess
    import tempfile
    Z = "/uufs/chpc.utah.edu/common/home/dbrown-group1/zifan"
    d = tempfile.mkdtemp(prefix="vllmgen_", dir=f"{Z}/graderdrift_runs/vllm_bench")
    json.dump(dict(model=POLICY, adapter=adapter if adapter and os.path.exists(adapter) else None,
                   prompts=texts, n=n, temperature=temperature, max_new=max_new, seed=seed),
              open(f"{d}/spec.json", "w"))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_LIBRARY_PATH")}
    env["PATH"] = f"{Z}/miniconda3/envs/vllmgen/bin:" + env.get("PATH", "")
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    for k, sub in (("XDG_CACHE_HOME", ""), ("TRITON_CACHE_DIR", "triton"),
                   ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("FLASHINFER_JIT_DIR", "flashinfer"),
                   ("VLLM_CACHE_ROOT", "vllm")):
        env[k] = os.path.join(f"{Z}/cache", sub)
        os.makedirs(env[k], exist_ok=True)
    t0 = time.time()
    r = subprocess.run([VLLM_PY, "vllm_gen.py", f"{d}/spec.json", f"{d}/out.json"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"vllm_gen failed rc={r.returncode}\n{r.stdout[-1500:]}\n"
                           f"{r.stderr[-2500:]}")
    resp = json.load(open(f"{d}/out.json"))["responses"]
    assert len(resp) == len(texts) * n, f"{len(resp)} != {len(texts)}x{n}"
    tick(f"  vllm: {len(texts)} prompts x {n} in {time.time()-t0:.0f}s "
         f"({(time.time()-t0)/len(texts):.3f} s/prompt)")
    return resp


# ---------------------------------------------------------------- generation
def gen_samples(adapter, prompts, n, temperature, max_new, batch, fewshot="", backend="hf"):
    user = [compose(q, fewshot) for q in prompts]
    if backend == "vllm":
        return gen_samples_vllm(adapter, user, n, temperature, max_new)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(POLICY, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(POLICY, torch_dtype=DTYPE, device_map="cuda")
    if adapter and os.path.exists(adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        tick(f"  policy + adapter {os.path.basename(adapter)}")
    model.eval()
    texts = [tok.apply_chat_template([{"role": "user", "content": u}],
                                     tokenize=False, add_generation_prompt=True) for u in user]
    # Every sequence in a batch is padded to the longest one, and MATH prompts vary a lot, so
    # unsorted batching spends much of its compute on padding. Sorting by length groups similar
    # prompts together; the original order is restored afterwards so the caller is unaffected.
    order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
    texts = [texts[j] for j in order]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            # 2048 not 1024: few-shot exemplars make the prompt several hundred tokens longer, and
            # left-truncation at 1024 would silently amputate them.
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to("cuda")
            g = model.generate(**enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                               top_p=0.95, num_return_sequences=n, max_new_tokens=max_new,
                               pad_token_id=tok.pad_token_id)
            g = g[:, enc["input_ids"].shape[1]:]
            out.extend(tok.batch_decode(g, skip_special_tokens=True))
            if (i // batch) % 10 == 0:
                tick(f"  generated {min(i+batch, len(texts))}/{len(texts)}")
    del model; free()
    # undo the length sort: out is prompt-major with n consecutive samples per prompt
    restored = [None] * len(out)
    for pos, orig in enumerate(order):
        restored[orig * n:(orig + 1) * n] = out[pos * n:(pos + 1) * n]
    return restored


# ---------------------------------------------------------------- grader + reward model
# Gated reward models addressable as graders or as the gold. The dimension is the number of
# criteria the model exposes, and it is NOT constant across a vendor's own releases: QRM shipped 19
# objectives in v1 (2024-09-25) and 5 in v2 (2024-11-05), six weeks apart, same architecture. That
# rubric cut is real observed phi-drift -- the criterion set changing size -- which is the case the
# formalism calls hard and which no synthetic w-rotation reproduces.
GRADERS = {
    "armorm": ("RLHFlow/ArmoRM-Llama3-8B-v0.1", 19),
    "qrm_v1": ("nicolinho/QRM-Llama3.1-8B", 19),
    "qrm_v2": ("nicolinho/QRM-Llama3.1-8B-v2", 5),
}


def armo_features(prompts, resp, n):
    """Back-compat shim: the original ArmoRM-only path, used by task=math and task=gsm8k."""
    return grader_features("armorm", prompts, resp, n)


def grader_features(key, prompts, resp, n, return_score=False):
    """phi and the model's own untouched gating, for any gated reward model in GRADERS.

    Returns (phi, W) with last dim = that model's criterion count, so a caller must not assume 19.
    """
    import vintage_score as V
    D = GRADERS[key][1]
    # Do NOT pass dtype_key: the JUDGES registry records a per-model dtype for a reason. QRM v1's
    # gating network uses BatchNorm, which rejects bf16 with "expected scalar type Float but found
    # BFloat16", so its entry says fp32. Overriding that with the POLICY's dtype crashed the first
    # UltraFeedback smoke run. The grader's precision is independent of the policy's.
    model, tok, kind = V.load_judge(key)
    P = len(prompts)
    phi = np.zeros((P, n, D), np.float32); W = np.zeros((P, n, D), np.float32)
    sc = np.zeros((P, n), np.float32)
    for p in range(P):
        for j in range(n):
            s_, ph, gt = V.score_response(model, tok, kind, prompts[p], resp[p * n + j])
            assert ph is not None and gt is not None, f"{key} exposed no phi/gate"
            phi[p, j], W[p, j], sc[p, j] = ph[:D], gt[:D], s_
        if (p + 1) % 50 == 0:
            tick(f"  graded {p+1}/{P} with {key}")
    del model; free()
    return (phi, W, sc) if return_score else (phi, W)


# Index of `ultrafeedback-overall_score` in the shared 19-criterion vector. ArmoRM and QRM v1 list
# the identical 19 attributes in the identical order, so this index addresses the same criterion in
# either model.
UF_OVERALL_IDX = 5


def gold_scores(key, prompts, resp, head=None):
    """The STATIC evaluator's score for one response per prompt, continuous, so paired comparisons
    use signed ranks rather than McNemar.

    `head=None` returns the model's own gated score. `head=i` returns criterion i of phi instead,
    and that is the better default for a gold: the GATED score is not actually static as an
    instrument, because the gating weights are computed from the prompt and response, so as the
    policy's outputs move across rounds the "same" gold applies different weights. A single head is
    a fixed coordinate of phi and cannot drift with the policy. It also agrees with the
    UltraFeedback labels better than the gated score does (0.7978 vs 0.7773), widening the headroom
    over QRM v1 from +0.033 to +0.054.

    Caveat to carry into the write-up: the ultrafeedback-overall_score head is trained to predict the
    very label we validate against, so it is an in-distribution fit, not independent judgement. It
    inherits that label's noise and cannot be used to validate it.
    """
    import vintage_score as V
    # Do NOT pass dtype_key: the JUDGES registry records a per-model dtype for a reason. QRM v1's
    # gating network uses BatchNorm, which rejects bf16 with "expected scalar type Float but found
    # BFloat16", so its entry says fp32. Overriding that with the POLICY's dtype crashed the first
    # UltraFeedback smoke run. The grader's precision is independent of the policy's.
    model, tok, kind = V.load_judge(key)
    out = np.zeros(len(prompts), np.float32)
    gated = np.zeros(len(prompts), np.float32)
    for i, (q, r) in enumerate(zip(prompts, resp)):
        sc, ph, _ = V.score_response(model, tok, kind, q, r)
        gated[i] = sc
        if head is None:
            out[i] = sc
        else:
            assert ph is not None and len(ph) > head, \
                f"{key} exposes {None if ph is None else len(ph)} criteria; head {head} out of range"
            out[i] = ph[head]
        if (i + 1) % 100 == 0:
            tick(f"  gold-scored {i+1}/{len(prompts)} with {key}"
                 + (f" head {head}" if head is not None else " (gated)"))
    del model; free()
    # report both so the gated score is always available as a secondary readout
    return out, gated


def bt_fit(X, y, l2=1e-3, iters=600, lr=1.0):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
    return w


def fit_reward_model(phi, grader_score, rng, pairs_per_pool=12):
    """The reward model sees the grader's PREFERENCES, never its weights. It learns a global w from
    pairwise labels, which is the coarsest honest version of a fitted RM and the one whose drift
    transmission we measured."""
    P, n, D = phi.shape
    mu = phi.reshape(-1, D).mean(0); sd = phi.reshape(-1, D).std(0) + 1e-9
    Z = (phi - mu) / sd
    a = rng.integers(0, n, (P, pairs_per_pool)); b = (a + 1 + rng.integers(0, n - 1, (P, pairs_per_pool))) % n
    pi = np.repeat(np.arange(P), pairs_per_pool)
    X = Z[pi, a.ravel()] - Z[pi, b.ravel()]
    y = (grader_score[pi, a.ravel()] > grader_score[pi, b.ravel()]).astype(float)
    w = bt_fit(X, y)
    agree = float(((X @ w > 0) == (y > 0.5)).mean())
    return w, mu, sd, agree


# ---------------------------------------------------------------- policy update
def _seq_logp(model, ids, nprompt):
    """Summed log-probability of the RESPONSE tokens only, prompt masked out."""
    logits = model(input_ids=ids).logits[:, :-1]
    labels = ids[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(tok)
    mask[:, nprompt - 1:] = 1.0            # labels are shifted by one
    return (tok * mask).sum(-1)


def dpo_train(adapter_in, adapter_out, prompts, win, lose, epochs=3, lr=5e-6, beta=0.1,
              maxlen=1024, accum=32, warmup=0.03, clip=1.0):
    """Train on the MARGIN between the reward model's winner and loser, not on the winner alone.

    SFT on the winner has almost no gradient here: the policy generated that response, and we
    measured its loss at 0.032, meaning it already assigns ~97% probability to its own target.
    The gap between winner and loser, by contrast, starts near zero, because both came from the
    same sampler. That gap is where the reward model's judgement actually lives and where there is
    room to move. With LoRA the frozen reference is free -- disabling the adapter gives back the
    starting policy, so no second copy is loaded.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    tok = AutoTokenizer.from_pretrained(POLICY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(POLICY, torch_dtype=DTYPE, device_map="cuda")
    if adapter_in and os.path.exists(adapter_in):
        model = PeftModel.from_pretrained(model, adapter_in, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.train(); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr,
                            weight_decay=0.0)
    # Same defect the SFT path had: this stepped on every pair, i.e. effective batch 1. accum=32
    # rather than 64 because each DPO step costs four forward passes (policy and frozen reference,
    # winner and loser), so the gradient per step is already less noisy than SFT's.
    steps_total = max(1, (len(prompts) * epochs) // accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps_total, pct_start=warmup, anneal_strategy="cos",
        cycle_momentum=False, div_factor=10.0, final_div_factor=10.0)

    def encode(i, resp):
        msgs = [{"role": "user", "content": f"{prompts[i]}\n\n{TASK['instr']}"},
                {"role": "assistant", "content": resp}]
        full = tok.apply_chat_template(msgs, tokenize=False)
        pre = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
        ids = tok(full, return_tensors="pt", truncation=True, max_length=maxlen).input_ids.cuda()
        return ids, len(tok(pre, truncation=True, max_length=maxlen).input_ids)

    idx = np.arange(len(prompts))
    tot = acc = 0.0; used = skipped = 0
    pending = nsteps = 0
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(idx)
        for k, i in enumerate(idx):
            iw, nw = encode(i, win[i]); il, nl = encode(i, lose[i])
            if nw >= iw.shape[1] or nl >= il.shape[1]:
                skipped += 1; continue
            with torch.no_grad(), model.disable_adapter():
                rw = _seq_logp(model, iw, nw); rl = _seq_logp(model, il, nl)
            pw = _seq_logp(model, iw, nw); pl = _seq_logp(model, il, nl)
            margin = beta * ((pw - rw) - (pl - rl))
            loss = -torch.nn.functional.logsigmoid(margin).mean()
            if not torch.isfinite(loss):
                skipped += 1; continue
            (loss / accum).backward()
            pending += 1
            tot += float(loss); acc += float((margin > 0).float().mean()); used += 1
            if pending == accum:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad], clip)
                opt.step(); opt.zero_grad(set_to_none=True)
                if nsteps + 1 < steps_total:
                    sched.step()
                nsteps += 1; pending = 0
            if (k + 1) % 200 == 0:
                tick(f"  dpo ep{ep+1} {k+1}/{len(idx)} loss {tot/max(used,1):.3f} "
                     f"margin>0 {acc/max(used,1):.3f} steps {nsteps}")
        if pending:
            torch.nn.utils.clip_grad_norm_(
                [q for q in model.parameters() if q.requires_grad], clip)
            opt.step(); opt.zero_grad(set_to_none=True); nsteps += 1; pending = 0
    tick(f"  dpo done: {used} pairs, {nsteps} optimizer steps (effective batch {accum}), "
         f"{epochs} epoch(s), final margin>0 {acc/max(used,1):.3f}"
         + (f", skipped {skipped}" if skipped else ""))
    model.save_pretrained(adapter_out)
    del model, opt, sched
    free()
    return tot / max(used, 1)


def lora_train(adapter_in, adapter_out, prompts, targets, epochs=3, lr=1e-4, accum=64,
               maxlen=1024, warmup=0.03, clip=1.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    tok = AutoTokenizer.from_pretrained(POLICY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(POLICY, torch_dtype=DTYPE, device_map="cuda")
    if adapter_in and os.path.exists(adapter_in):
        model = PeftModel.from_pretrained(model, adapter_in, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            # QLoRA's finding is that LoRA has to cover the MLP, not just attention, to track full
            # fine-tuning. Attention-only was ~17M trainable params; adding gate/up/down makes it
            # ~40M at the same rank, for no extra activation memory.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # weight_decay=0.0 explicitly: torch's AdamW defaults to 0.01, which was being applied to the
    # LoRA parameters unintentionally. QLoRA uses 0.0.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=0.0)
    # Effective batch was 1: the old loop stepped on every example, i.e. ~1000 noisy updates per
    # epoch at lr=1e-4 with no clipping and no warmup, against RFT's batch of 128. Accumulating to
    # `accum` costs no extra memory -- the micro-batch is still one example, so activation memory is
    # unchanged; only the moment when opt.step() fires changes.
    steps_total = max(1, (len(prompts) * epochs) // accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps_total, pct_start=warmup, anneal_strategy="cos",
        cycle_momentum=False, div_factor=10.0, final_div_factor=10.0)
    idx = np.arange(len(prompts))
    skipped = used = 0
    pending = 0          # micro-batches accumulated since the last optimizer step
    nsteps = 0
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(idx)
        tot = 0.0
        for k, i in enumerate(idx):
            msgs = [{"role": "user", "content": f"{prompts[i]}\n\n{TASK['instr']}"},
                    {"role": "assistant", "content": targets[i]}]
            full = tok.apply_chat_template(msgs, tokenize=False)
            pre = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
            ids = tok(full, return_tensors="pt", truncation=True, max_length=maxlen).input_ids.cuda()
            nprompt = len(tok(pre, truncation=True, max_length=maxlen).input_ids)
            lab = ids.clone(); lab[:, :nprompt] = -100          # loss on the response only
            if nprompt >= ids.shape[1]:
                # The prompt filled maxlen, so truncation left no response tokens and every label
                # is -100. Cross-entropy over an empty target is nan; the gradient is zero, so the
                # step is harmless, but the nan poisons the reported loss for the whole epoch.
                # 2 of 1000 MATH training prompts hit this.
                skipped += 1
                continue
            loss = model(input_ids=ids, labels=lab).loss
            if not torch.isfinite(loss):
                skipped += 1
                continue
            # scale so the accumulated gradient is the MEAN over the effective batch, not the sum
            (loss / accum).backward()
            pending += 1
            tot += float(loss); used += 1
            if pending == accum:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad], clip)
                opt.step(); opt.zero_grad(set_to_none=True)
                if nsteps + 1 < steps_total:
                    sched.step()
                nsteps += 1; pending = 0
            if (k + 1) % 200 == 0:
                tick(f"  sft ep{ep+1} {k+1}/{len(idx)} loss {tot/max(used,1):.3f} "
                     f"steps {nsteps} lr {opt.param_groups[0]['lr']:.2e}")
        # a partial final batch still carries signal; step on it rather than discarding it
        if pending:
            torch.nn.utils.clip_grad_norm_(
                [q for q in model.parameters() if q.requires_grad], clip)
            opt.step(); opt.zero_grad(set_to_none=True); nsteps += 1; pending = 0
    tick(f"  sft done: {used} examples, {nsteps} optimizer steps "
         f"(effective batch {accum}), {epochs} epoch(s)"
         + (f", skipped {skipped} with no trainable target tokens" if skipped else ""))
    model.save_pretrained(adapter_out)
    # `sched` holds the optimizer, which holds the LoRA parameters, which hold the model. Deleting
    # only `model` and `opt` left ~13 GiB resident, which starved the vLLM subprocess launched next
    # and killed two 2-hour runs on "Free memory on device cuda:0 (31.19/44.42 GiB) on startup is
    # less than desired GPU memory utilization".
    del model, opt, sched
    free()
    return tot / max(used, 1)


# ---------------------------------------------------------------- evaluation
def evaluate(adapter, prompts, golds, max_new, batch, backend="hf"):
    """Greedy decode on HELD-OUT problems, scored against MATH's exact answer key.

    Returns PER-PROBLEM outcomes, not just the mean. Every arm sees the identical held-out set, so
    arms can be compared pairwise, and a paired comparison is far more sensitive than comparing two
    means: its error depends on how often the arms DISAGREE, not on the base rate. Two runs of an
    identical configuration swung 0.040 apart on 200 problems, which is larger than any effect we
    are looking for, and that swing is mostly shared noise a paired test removes."""
    # Evaluation is ZERO-SHOT even when training candidates are elicited few-shot (STaR): the point
    # is whether the policy internalised the behaviour without exemplars.
    resp = gen_samples(adapter, prompts, 1, 0.0, max_new, batch, fewshot="", backend=backend)
    if TASK["key"] == "uf":
        # Non-verifiable task: the gold is a STATIC reward model, better than the grader by a
        # measured margin (ArmoRM 0.7773 vs QRM v1 0.7439 against the GPT-4 labels). Its score is
        # continuous, so `hit` holds real values and paired tests use signed ranks, not McNemar.
        hit, gated = gold_scores(TASK["gold_model"], prompts, resp, head=TASK["gold_head"])
        return float(hit.mean()), hit, dict(
            fmt_rate=float("nan"), parse_rate=1.0,
            mean_chars=float(np.mean([len(r) for r in resp])),
            gold_model=TASK["gold_model"], gold_head=TASK["gold_head"],
            gold_gated_mean=float(gated.mean()), gold_gated=gated.tolist())
    pred, same = TASK["pred"], TASK["same"]
    hit = np.array([float(same(pred(resp[i]), golds[i])) for i in range(len(prompts))])
    # Accuracy alone cannot distinguish "reasons worse" from "stopped emitting a parseable answer".
    # gsm8k_pred falls back to the LAST NUMBER in the text when the '#### n' marker is missing, so a
    # formatting regression silently becomes a wrong answer instead of an obvious failure. Track the
    # marker rate and the response length (the standard Goodhart channel) alongside the accuracy.
    marker = "####" if TASK["key"] == "gsm8k" else "\\boxed"
    fmt = float(np.mean([marker in r for r in resp]))
    parsed = float(np.mean([pred(r) is not None for r in resp]))
    chars = float(np.mean([len(r) for r in resp]))
    return float(hit.mean()), hit, dict(fmt_rate=fmt, parse_rate=parsed, mean_chars=chars)


def main():
    global DTYPE, POLICY          # must precede any read of POLICY, incl. as an argparse default
    ap = argparse.ArgumentParser()
    # release_swap_real: drift by swapping to a genuinely released successor rather than rotating w.
    ap.add_argument("--arm", required=True,
                    choices=["stationary", "release_swap", "matched_random", "release_swap_real",
                             "satiation", "self_preference", "acclimation", "oracle",
                             "habituation",  # deprecated alias for acclimation; see drift_w
                             "momentum"])
    # Endogenous intensity. Defaults None -> the module constants, so omitting them reproduces
    # every arm run before these flags existed. eta drives satiation/assimilation/momentum;
    # lam and alpha drive acclimation. Calibrated on the round-0 pool: satiation eta 0.25/0.32/0.40
    # gives flip 0.147/0.275/0.569, acclimation lam 1/3/9 gives 0.003/0.019/0.294, assimilation
    # saturates at 0.155 because it cannot rotate past the profile direction (58.4 deg from w0).
    ap.add_argument("--drift-cumulative", action="store_true",
                    help="endogenous arms blend from the PREVIOUS drifted direction instead of from "
                         "the current undrifted w0, so displacement accumulates across rounds")
    ap.add_argument("--drift-eta", type=float, default=None)
    ap.add_argument("--drift-lam", type=float, default=None)
    ap.add_argument("--drift-alpha", type=float, default=None)
    ap.add_argument("--drift-schedule", default="oneshot",
                    choices=["oneshot", "ramp", "oscillate", "walk"],
                    help="how the exogenous rotation is delivered over rounds; oneshot reproduces "
                         "every run made before this option existed")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--drift-round", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=500)   # all of MATH-500, disjoint from train
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--policy-step", choices=["sft", "dpo"], default="sft")
    # Which reward model sits between the grader and the policy, if any.
    #   linear : global Bradley-Terry fit on ArmoRM's 19 criteria. Cheap, and the low-transmission
    #            end: it cannot represent a prompt-conditional rubric, so after a 36.4 deg drift it
    #            reproduces the grader's own pick only 45% of the time (0.89 undrifted).
    #   gate   : prompt-conditional MLP, matching ArmoRM's actual functional form.
    #   none   : NO fitted layer. The grader's own argmax/argmin become the preference pair and DPO
    #            trains on them directly, which is DPO's original reason for existing. This is the
    #            upper bound on drift transmission and the control the other two are read against:
    #            if drift cannot hurt at 100% transmission, the fitting layer is beside the point.
    ap.add_argument("--reward-model", choices=["linear", "gate", "none"], default="linear")
    ap.add_argument("--dpo-lr", type=float, default=5e-6)
    # DPO's implicit KL strength against the FROZEN BASE policy (disable_adapter), not
    # against the previous round. Was hardcoded at 0.1, which is too loose for GSM8K: the
    # stationary arm ran its mean output from 664 to 1946 characters in one round, hit the
    # 400-token cap mid-derivation, and accuracy fell 0.784 -> 0.108 with parse rate still
    # 1.000. UltraFeedback shows the same growth, 909 -> 1330, but its gold rewards length
    # (corr +0.511 across stationary round-points) so it never surfaces as harm.
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--task", choices=["math", "gsm8k", "uf"], default="math")
    # Grader and gold are now separate models. On UltraFeedback the only configuration where the
    # gold is measurably BETTER than the grader is QRM as grader with ArmoRM as gold (0.7439 vs
    # 0.7773 against the GPT-4 labels) -- the first non-zero headroom found in this project.
    ap.add_argument("--grader", choices=sorted(GRADERS), default="armorm")
    ap.add_argument("--grader-drifted", choices=sorted(GRADERS), default="qrm_v2",
                    help="the released successor the grader becomes in arm release_swap_real")
    ap.add_argument("--gold-model", choices=sorted(GRADERS), default="armorm",
                    help="static evaluator for task=uf; must be better than --grader")
    # Default to the ultrafeedback-overall_score head rather than the gated score: the gating is
    # computed from the response, so the gated score is not a static instrument across rounds.
    ap.add_argument("--gold-head", type=int, default=UF_OVERALL_IDX,
                    help="criterion index of phi to use as the gold; -1 for the gated score")
    # --train-source applies to math only; gsm8k always uses its own disjoint train split.
    ap.add_argument("--train-source", choices=["math500", "math_train"], default="math500")
    # 3 epochs and an effective batch of 64 follow RFT (3 epochs, batch 128). The previous defaults
    # were 1 epoch at effective batch 1.
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=64)
    # 8-shot is the standard GSM8K prompt in the literature; exemplars are excluded from training.
    ap.add_argument("--n-shot", type=int, default=8)
    ap.add_argument("--gold-filter", action="store_true",
                    help="RFT-style: train only on pools containing at least one correct sample")
    ap.add_argument("--gen-backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--drift-deg", type=float, default=None,
                    help="rotation magnitude for the exogenous arms; default MEASURED_DEG=36.4, "
                         "which was calibrated on an ArmoRM release pair. On UltraFeedback use ~45 "
                         "to match the real QRM v1->v2 selection-flip rate of 27%%.")
    ap.add_argument("--resume", action="store_true",
                    help="reuse adapter_r<n> from a previous attempt instead of retraining it")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    POLICY = a.policy
    DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    if a.dtype == "fp32":
        # On Hopper and Ampere, torch may route fp32 matmuls through TF32 tensor cores, which keeps
        # fp32 storage but drops the mantissa to 10 bits. That is neither fp32 nor reproducible
        # across architectures, so it would defeat the reason for running fp32 at all.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    if a.smoke:
        a.rounds, a.n_train, a.n_eval, a.n_samples, a.max_new = 2, 24, 24, 4, 320
    os.makedirs(a.out, exist_ok=True)

    # Prompt count is the lever prior work actually pulls. RAFT runs 20k prompts per iteration and
    # keeps only the top-1 of K=4; RFT runs 7.5k questions. Our first configuration used 300 prompts
    # from MATH-500 and the policy did not move at all. MATH's own 12k training split is disjoint
    # from MATH-500 (verified: 0 of 500 overlap), so training and evaluation stay clean.
    from datasets import load_dataset

    import bon_pilot as BP
    rng = np.random.default_rng(a.seed)
    if a.task == "uf":
        # Non-verifiable chat task, and the only configuration in this project where the gold is
        # measurably BETTER than the grader: against the UltraFeedback GPT-4 labels ArmoRM scores
        # 0.7773 and QRM v1 scores 0.7439. Prompts come from the same pool the reward-model panel was
        # measured on, so those agreement numbers apply to exactly these prompts. There is no answer
        # key -- the gold is a static reward model and its score is continuous.
        assert a.gold_model != a.grader, \
            f"gold and grader are both {a.grader}; the gold must be an independent model"
        import uf_gold
        qs, _, _, _ = uf_gold.load_pool(None, None)
        TASK.update(key="uf", instr="", pred=None, same=None, gold_model=a.gold_model,
                    gold_head=(None if a.gold_head < 0 else a.gold_head))
        perm = rng.permutation(len(qs))
        tr, ev = perm[: a.n_train], perm[a.n_train: a.n_train + a.n_eval]
        trQ = [qs[int(i)] for i in tr]
        evQ = [qs[int(i)] for i in ev]
        trG = [None] * len(trQ)          # no answer key exists on a non-verifiable task
        evG = [None] * len(evQ)
        assert len(trQ) and len(evQ), f"n_train+n_eval exceeds the {len(qs)}-prompt pool"
        tick(f"task=uf: {len(qs)} pool prompts -> {len(trQ)} train / {len(evQ)} held out, disjoint")
        tick(f"  grader={a.grader} ({GRADERS[a.grader][1]} criteria)  "
             f"gold={a.gold_model} ({GRADERS[a.gold_model][1]} criteria)"
             + (f" head {a.gold_head}=ultrafeedback-overall_score" if a.gold_head >= 0
                else " gated score"))
        return _run_rounds(a, trQ, trG, evQ, evG, "")
    # Reuse bon_pilot's answer extractor and matcher instead of restating them here. The best-of-n
    # pools that chose the base policy were scored with exactly these, so any divergence would make
    # stage 1's accuracy quietly incomparable to the pool it was selected from.
    BT = BP.TASKS[a.task]
    TASK.update(key=a.task, pred=BT["pred"], same=BT["same"],
                # MATH keeps stage 1's own wording, which differs from bon_pilot's; every earlier
                # MATH run used it and changing it would break comparability. GSM8K has no prior
                # stage-1 run to match, so it takes bon_pilot's, which is what generated its pools.
                instr=MATH_INSTR if a.task == "math" else BT["instr"])
    if a.task == "gsm8k":
        # GSM8K ships disjoint train (7473) and test (1319) splits, so unlike MATH there is no
        # train/eval overlap to verify by hand.
        ev_ds = load_dataset("openai/gsm8k", "main", split="test")
        tr_ds = load_dataset("openai/gsm8k", "main", split="train")
        # Exemplars come from the first n_shot train rows and those rows are then excluded from the
        # training prompts, so an exemplar is never also a training question.
        pool_idx = rng.permutation(np.arange(a.n_shot, len(tr_ds)))
        tr = pool_idx[: a.n_train]
        ev = rng.permutation(len(ev_ds))[: a.n_eval]
    elif a.train_source == "math_train":
        ev_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        tr_ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
        tr = rng.permutation(len(tr_ds))[: a.n_train]
        ev = rng.permutation(len(ev_ds))[: a.n_eval]
    else:                                   # original: carve both out of MATH-500
        ev_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        tr_ds = ev_ds
        perm = rng.permutation(len(ev_ds))
        tr, ev = perm[: a.n_train], perm[a.n_train: a.n_train + a.n_eval]
    qf, gf = BT["q"], BT["gold"]

    def pull(ds, idx):
        rows = [ds[int(i)] for i in idx]
        return ([r[qf] if isinstance(qf, str) else qf(r) for r in rows], [gf(r) for r in rows])

    trQ, trG = pull(tr_ds, tr)
    evQ, evG = pull(ev_ds, ev)
    assert all(g is not None for g in trG + evG), "gold extraction returned None"

    # STaR / RFT bootstrapping: few-shot prompts are used ONLY to generate training candidates, and
    # never at evaluation. The policy is then fine-tuned on the bare zero-shot prompt paired with a
    # few-shot-elicited response, so improvement shows up as the model internalising the format and
    # the reasoning without exemplars. This matters concretely here: tulu-2 emitted the '####' marker
    # on 4.2% of zero-shot responses, so every correctness score was riding gsm8k_pred's
    # last-number-in-the-text fallback.
    fewshot = ""
    if a.n_shot and a.task == "gsm8k":
        ex = [tr_ds[int(i)] for i in range(a.n_shot)]
        fewshot = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in ex)
        tick(f"few-shot: {a.n_shot} exemplars, {len(fewshot)} chars, "
             f"excluded from the {len(trQ)} training prompts")
    elif a.n_shot:
        tick(f"few-shot requested but not defined for task={a.task}; running zero-shot")
    return _run_rounds(a, trQ, trG, evQ, evG, fewshot)


def _run_rounds(a, trQ, trG, evQ, evG, fewshot):
    """The round loop, shared by every task.

    Extracted from main() so the verifiable tasks (math, gsm8k) and the non-verifiable one (uf) run
    byte-identical machinery -- generation, grading, selection, the gold filter, the policy update,
    evaluation and logging. Only how the data was assembled differs, and that stays in main().
    `trG`/`evG` are None throughout on a non-verifiable task; nothing downstream may assume an
    answer key exists without checking TASK["key"].
    """
    tick(f"policy={a.policy} step={a.policy_step} reward_model={a.reward_model}")
    tick(f"dtype={a.dtype} gpu={torch.cuda.get_device_name(0)} "
         f"task={a.task} source={a.train_source if a.task == 'math' else a.task} "
         f"epochs={a.epochs} lr={a.lr} max_new={a.max_new}")
    tick(f"n_shot={a.n_shot} gold_filter={a.gold_filter} backend={a.gen_backend} "
         f"accum={a.accum} epochs={a.epochs}")
    tick(f"arm={a.arm} rounds={a.rounds} train={len(trQ)} heldout={len(evQ)} "
         f"drift at round {a.drift_round if a.arm!='stationary' else 'never'}")

    hist = []
    last_gkey = None
    profile = None        # standardised criterion profile of what the grader has selected so far
    acc0, hit0, ev0 = evaluate(None, evQ, evG, a.max_new, a.batch, backend=a.gen_backend)
    if TASK["key"] == "uf":
        tick(f"round 0 (untrained policy): gold RM score {acc0:.4f} from {TASK['gold_model']} "
             f"on {len(evQ)} held-out prompts; mean {ev0['mean_chars']:.0f} chars")
    else:
        tick(f"round 0 (untrained policy): held-out accuracy {acc0:.3f} on {len(evQ)} problems; "
             f"format {ev0['fmt_rate']:.3f} parse {ev0['parse_rate']:.3f} "
             f"mean {ev0['mean_chars']:.0f} chars")
    hist.append(dict(round=0, heldout_acc=acc0, per_problem=hit0.tolist(), **ev0))
    adapter = None
    for r in range(1, a.rounds + 1):
        tick(f"===== round {r} =====")
        ad_r = os.path.join(a.out, f"adapter_r{r}")
        if a.resume and os.path.isdir(ad_r):
            # A previous attempt finished this round's training and saved the adapter, then died
            # before writing history (the vLLM-after-SFT memory crash). Re-evaluating the saved
            # adapter costs minutes; regenerating and retraining the round costs ~90.
            # Only safe for arms that do not consume `profile`, which accumulates across rounds from
            # the responses the grader selected and cannot be reconstructed from an adapter alone.
            # release_swap_real belongs here too: drift_w returns w0 unchanged for it, so its
            # drift comes entirely from swapping the released grader model and it never reads
            # `profile`. Omitting it was an oversight from when the arm was added, and it turned a
            # preemption into an AssertionError that discarded a completed round.
            assert a.arm in ("stationary", "oracle", "release_swap", "matched_random",
                             "release_swap_real"), \
                f"--resume cannot reconstruct the criterion profile that arm {a.arm} depends on"
            adapter = ad_r
            acc, hit, evs = evaluate(adapter, evQ, evG, a.max_new, a.batch, backend=a.gen_backend)
            # sign test on the paired difference: works for binary correctness and for a
            # continuous reward-model gold alike
            _d = hit - hit0
            n_up, n_dn = int((_d > 0).sum()), int((_d < 0).sum())
            tick(f"  RESUMED round {r} from {os.path.basename(ad_r)}: held-out accuracy {acc:.3f} "
                 f"vs round 0: {n_up} fixed, {n_dn} broken; format {evs['fmt_rate']:.3f}")
            hist.append(dict(round=r, heldout_acc=acc, per_problem=hit.tolist(),
                             n_fixed=n_up, n_broken=n_dn, resumed=True, **evs))
            json.dump(dict(arm=a.arm, args=vars(a), history=hist),
                      open(os.path.join(a.out, "history.json"), "w"), indent=2)
            continue
        resp = gen_samples(adapter, trQ, a.n_samples, a.temperature, a.max_new, a.batch,
                           fewshot=fewshot, backend=a.gen_backend)
        # Which grader is in force this round. For the real-drift arm the grader is a DIFFERENT
        # RELEASED MODEL after drift_round -- not a rotation of the same weights -- so phi itself
        # changes dimension (QRM v1's 19 criteria to v2's 5). That is the phi-drift channel.
        gkey = (a.grader_drifted if (a.arm == "release_swap_real" and r >= a.drift_round)
                else a.grader)
        if gkey != last_gkey:
            tick(f"  grader in force: {gkey} ({GRADERS[gkey][1]} criteria)"
                 + ("  <-- SWAPPED, real released update" if r >= a.drift_round
                    and a.arm == "release_swap_real" else ""))
            last_gkey = gkey
        phi, W, gsc = grader_features(gkey, trQ, resp, a.n_samples, return_score=True)
        # The loop selects through the linear model u = w.phi. That is exact for QRM but NOT for
        # ArmoRM, whose score is its gating applied to CALIBRATED objectives while `rewards`
        # exposes raw ones: the reconstruction reproduces only about 15% of ArmoRM's own picks and
        # gives up 0.10 of gold before any drift. Running an arm on a grader that fails this check
        # trains against a surrogate nobody deploys, so refuse rather than record it.
        _fid = float((np.einsum("pnd,pd->pn", phi, W.mean(1)).argmax(1) == gsc.argmax(1)).mean())
        if r == 1:
            tick(f"  grader fidelity: the linear model reproduces {_fid:.3f} of {gkey}'s own picks")
        assert _fid >= 0.90, (
            f"{gkey}: u = w.phi reproduces only {_fid:.1%} of this grader's own picks. The loop "
            f"would train against a linear surrogate, not against {gkey}.")
        w0 = W.mean(1)
        wg = drift_w(w0, a.arm, r, a.drift_round, seed=a.seed + 17, profile=profile,
                     deg=a.drift_deg, schedule=a.drift_schedule,
                     eta=a.drift_eta, lam=a.drift_lam, alpha=a.drift_alpha,
                     cumulative=a.drift_cumulative)
        gscore = np.einsum("pnd,pd->pn", phi, wg)              # the GRADER's opinion
        # The drift-induced FLIP RATE: how many selections this round's weights move relative to the
        # undrifted grader on the same pool. This is the quantity the selection-level study found
        # explains 64-99% of harm, and without it a loop arm reports an outcome with no measure of
        # how much drift actually reached the selection step. It also makes the exposure prediction
        # testable: the effective dimension of the criterion frame RISES with training, so the same
        # angle should flip progressively more picks in later rounds.
        _sel_undrifted = np.einsum("pnd,pd->pn", phi, w0).argmax(1)
        drift_flip = float((gscore.argmax(1) != _sel_undrifted).mean())
        moved = float(np.median(np.degrees(np.arccos(
            np.clip((unit(w0) * unit(wg)).sum(1), -1, 1)))))
        if a.reward_model == "none":
            rm_score, agree = gscore, 1.0          # the grader IS the selector; no fit involved
        else:
            wrm, mu, sd, agree = fit_reward_model(phi, gscore,
                                                  np.random.default_rng(a.seed + r))
            rm_score = np.einsum("pnd,d->pn", (phi - mu) / sd, wrm)
        sel = rm_score.argmax(1)
        # Correctness of every sampled response, computed for ALL arms. In non-oracle arms this is
        # MONITORING ONLY and never touches selection: it is the mediating variable the whole thesis
        # rests on (drift -> the grader picks worse targets -> the policy gets worse), and without it
        # a harm result has no mechanism attached to it.
        if TASK["key"] == "uf":
            # No answer key exists on a non-verifiable task. The oracle control and RFT's gold
            # filter both require one, so refuse them loudly rather than silently degrade: a run
            # that quietly skipped the filter would look like a filtered run in the logs.
            assert a.arm != "oracle", "the oracle arm needs an answer key; unavailable on task=uf"
            assert not a.gold_filter, "--gold-filter needs an answer key; unavailable on task=uf"
            ok_all = np.full((len(trQ), a.n_samples), np.nan, np.float32)
            pool_live = pool_dead = pool_all_ok = float("nan")
        else:
            _pred, _same = TASK["pred"], TASK["same"]
            ok_all = np.array([[float(_same(_pred(resp[p * a.n_samples + j]), trG[p]))
                                for j in range(a.n_samples)] for p in range(len(trQ))])
            pool_live = float(((ok_all.sum(1) > 0) & (ok_all.sum(1) < a.n_samples)).mean())
            pool_dead = float((ok_all.max(1) == 0).mean())
            pool_all_ok = float((ok_all.min(1) == 1).mean())
        if a.arm == "oracle":
            # POSITIVE CONTROL, not a drift arm. Select by the answer key instead of by the reward
            # model, which is what RFT does and what it reports +5.8 points from. If this improves
            # the policy and the reward-model arms do not, the bottleneck is selection quality: the
            # RM picks a correct response 68.6% of the time against 100% here. If this also fails,
            # the training setup is broken and no grader would rescue it. The gold is used ONLY
            # here and never in an arm whose result is reported as a drift measurement.
            ok = ok_all
            tie = rm_score - 1e6 * (1.0 - ok)      # prefer correct; break ties by the RM
            sel = tie.argmax(1)
            tick(f"  ORACLE arm: {float(ok.max(1).mean()):.3f} of pools have >=1 correct; "
                 f"selected response is correct in {float(ok[np.arange(len(trQ)), sel].mean()):.3f}")
        tick(f"  drift flip rate {drift_flip:.3f} of pools (vs the undrifted grader, same pool)")
        tick(f"  rm={a.reward_model}; grader moved {moved:.1f} deg; "
             f"RM reproduces grader pairs at {agree:.3f}; "
             f"RM and grader agree on the pick in "
             f"{float((sel == gscore.argmax(1)).mean()):.3f} of pools")
        # Selection quality: is the TRAINING TARGET correct? This is what drift should damage.
        sel_ok = float(ok_all[np.arange(len(trQ)), sel].mean())
        # Regret separates "the grader ranked badly" from "the pool had nothing good in it":
        # how much grader score it left on the table relative to the best response available.
        g_sel = gscore[np.arange(len(trQ)), sel]
        g_rng = gscore.max(1) - gscore.min(1) + 1e-9
        regret = float(((gscore.max(1) - g_sel) / g_rng).mean())
        tick(f"  pools: live {pool_live:.3f} dead {pool_dead:.3f} all-correct {pool_all_ok:.3f}; "
             f"TARGET CORRECT {sel_ok:.3f}; grader regret {regret:.3f}")
        # Never hardcode the criterion count. A real vendor update can change it: QRM v1 exposes 19
        # objectives and v2 exposes 5, so phi's last dimension is a property of whichever grader is
        # in force this round, not a constant.
        D = phi.shape[-1]
        mu_all = phi.reshape(-1, D).mean(0); sd_all = phi.reshape(-1, D).std(0) + 1e-9
        seen = ((phi[np.arange(len(trQ)), sel] - mu_all) / sd_all).mean(0)
        if profile is not None and profile.shape != seen.shape:
            # The rubric itself changed size, so the accumulated criterion profile is not comparable
            # to the new one and cannot be averaged with it. This is the identifiability problem the
            # formalism describes: when phi changes, quantities indexed by criteria lose their
            # meaning across the change. Reset rather than silently mix two coordinate systems.
            tick(f"  criterion count changed {profile.shape[0]} -> {seen.shape[0]}; "
                 f"resetting the accumulated criterion profile (not comparable across rubrics)")
            profile = None
        profile = seen if profile is None else 0.5 * profile + 0.5 * seen
        # RFT's filtering step, which we previously lacked. Without it a target is taken from EVERY
        # pool, including those where no sample is correct, so the policy is fine-tuned on wrong
        # arithmetic -- 34.4% of targets in the first GSM8K run. RFT and STaR both keep only correct
        # samples. The filter is gold-based but it is data CURATION, applied identically in every arm
        # and fixed before the grader acts; the grader still chooses which response inside a kept
        # pool becomes the target, which is where drift does its damage.
        if not a.gold_filter:
            keep = np.arange(len(trQ))
        elif a.policy_step == "dpo":
            # DPO trains on a CONTRAST, so a pool must contain both a correct and an incorrect
            # sample. Keeping merely ">=1 correct" would admit all-correct pools, where the rejected
            # response is also correct and the update pushes the policy away from a right answer --
            # roughly 10% of pools at round 2.
            keep = np.where((ok_all.max(1) > 0) & (ok_all.min(1) == 0))[0]
        else:
            keep = np.where(ok_all.max(1) > 0)[0]
        if a.gold_filter:
            # The two filters drop different things, so the message must say which. SFT keeps pools
            # with >=1 correct, so it drops only DEAD pools. DPO needs a contrast and so keeps LIVE
            # pools only, dropping dead AND all-correct ones -- on LLaMA3-SFT/GSM8K that was 44.1%
            # dropped, of which only 2.9 points were dead and 41.2 were all-correct. Reporting that
            # as "had no correct sample" was wrong and inverted the reader's picture of the pool.
            why = ("no correct sample" if a.policy_step != "dpo"
                   else "not live: no correct sample, or every sample correct")
            tick(f"  gold filter ({a.policy_step}): {len(keep)}/{len(trQ)} pools kept "
                 f"({1 - len(keep)/max(len(trQ),1):.1%} dropped -- {why}); "
                 f"target correct among kept {float(ok_all[keep, sel[keep]].mean()):.3f}")
        tgtQ = [trQ[p] for p in keep]
        tgt = [resp[p * a.n_samples + sel[p]] for p in keep]
        if not len(keep):
            # Every pool was dead. Training on nothing would divide by zero in the LR schedule; skip
            # the round and carry the previous adapter forward so the run still reports a trajectory.
            tick("  WARNING: no pool contained a correct sample; skipping this round's training")
            acc, hit, evs = evaluate(adapter, evQ, evG, a.max_new, a.batch, backend=a.gen_backend)
            hist.append(dict(round=r, heldout_acc=acc, per_problem=hit.tolist(), skipped=True,
                             n_train_kept=0, pool_dead=pool_dead, **evs))
            continue
        out_ad = os.path.join(a.out, f"adapter_r{r}")
        if a.policy_step == "dpo":
            worst = rm_score.argmin(1)
            if a.arm == "oracle":
                # The positive control has to contrast a CORRECT response with an INCORRECT one.
                # The grader's argmin inside a live pool is sometimes itself correct, which would
                # make the pair (correct, correct) and teach nothing. Push correct responses to the
                # top so argmin lands on an incorrect one, breaking ties by grader score as before.
                worst = (rm_score + 1e6 * ok_all).argmin(1)
                tick(f"  ORACLE dpo pair: chosen correct "
                     f"{float(ok_all[keep, sel[keep]].mean()):.3f}, "
                     f"rejected correct {float(ok_all[keep, worst[keep]].mean()):.3f} "
                     f"(the second should be ~0)")
            los = [resp[p * a.n_samples + worst[p]] for p in keep]
            loss = dpo_train(adapter, out_ad, tgtQ, tgt, los, epochs=a.epochs, lr=a.dpo_lr,
                             beta=a.dpo_beta,
                              accum=max(1, a.accum // 2))
        else:
            loss = lora_train(adapter, out_ad, tgtQ, tgt, epochs=a.epochs, lr=a.lr,
                              accum=a.accum)
        adapter = out_ad
        acc, hit, evs = evaluate(adapter, evQ, evG, a.max_new, a.batch, backend=a.gen_backend)
        d = hit - hit0
        n_up, n_dn = int((d > 0).sum()), int((d < 0).sum())
        _lab = "gold RM score" if TASK["key"] == "uf" else "held-out accuracy"
        _updn = "better, {} worse" if TASK["key"] == "uf" else "fixed, {} broken"
        tick(f"  round {r}: {_lab} {acc:.4f}  (loss {loss:.3f})  "
             f"vs round 0: {n_up} " + _updn.format(n_dn) +
             f", {len(hit)-n_up-n_dn} unchanged")
        # Exact McNemar on the paired outcomes, so a round-to-round change arrives with a p-value
        # instead of raw counts. The round-1 GSM8K numbers (49 fixed / 62 broken) look like a decline
        # and are not one: p = 0.255.
        nd = n_up + n_dn
        pval = min(1.0, 2.0 * sum(comb(nd, t) for t in range(min(n_up, n_dn) + 1)) / 2.0 ** nd) \
            if nd else 1.0
        hist.append(dict(round=r, heldout_acc=acc, per_problem=hit.tolist(),
                         n_fixed=n_up, n_broken=n_dn, mcnemar_p=pval,
                         grader_deg=moved, rm_pair_agree=agree,
                         rm_grader_pick_agree=float((sel == gscore.argmax(1)).mean()),
                         target_correct=sel_ok, grader_regret=regret, drift_flip=drift_flip,
                         pool_live=pool_live, pool_dead=pool_dead, pool_all_correct=pool_all_ok,
                         n_train_kept=int(len(keep)), n_shot=a.n_shot,
                         gen_backend=a.gen_backend,
                         sft_loss=loss, **evs))
        json.dump(dict(arm=a.arm, args=vars(a), history=hist),
                  open(os.path.join(a.out, "history.json"), "w"), indent=2)
    tick("done: " + " -> ".join(f"{h['heldout_acc']:.3f}" for h in hist))


if __name__ == "__main__":
    main()
