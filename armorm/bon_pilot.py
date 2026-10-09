"""Go/no-go for the ArmoRM port: does ArmoRM's ranking pick CORRECT answers?

The planned experiment measures harm as a fall in verifiable correctness when the grader drifts.
That only works if the undrifted grader selects for correctness to begin with, and there is real
doubt that it does. ArmoRM's correctness and truthfulness heads contribute approximately zero to its
output, its effective width is 2 to 4 of 19, and its nineteen criteria have a mean pairwise
correlation of +0.688 on real responses, so it may be ranking general fluency rather than whether
the arithmetic is right.

This samples n answers per GSM8K problem, scores every one with the real ArmoRM, checks each answer
against the gold, and compares four selection rules on the same pools:

  random      the accuracy of a uniformly drawn sample = the base model's accuracy
  best-of-n   argmax of ArmoRM's score
  worst-of-n  argmin, which bounds how much signal the ranking carries at all
  oracle      the share of problems where at least one sample is right = the headroom

If best-of-n does not beat random, the correctness channel is dead and the port needs a different
gold. If it does, the gap between best-of-n and oracle is the room a drift has to destroy.

Saves phi, the per-prompt gating weights w, scores and correctness for every sample, so the drift
arms can be replayed later without regenerating or rescoring anything.

Usage: python bon_pilot.py --gen MODEL --out FILE [--n-prompts 300] [--n-samples 8]
"""
import argparse
import os, json, os, re, time

import numpy as np
import torch

T0 = time.time()
def tick(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

GSM8K_INSTR = ("Solve the problem step by step. Finish your reply with the final numeric answer on "
               "its own last line, in the form '#### <number>'.")
MATH_INSTR = ("Solve the problem step by step. Put the final answer in \\boxed{}.")


def boxed(text):
    """Content of the LAST \\boxed{...}, with balanced braces."""
    k = text.rfind("\\boxed")
    if k < 0:
        return None
    i = text.find("{", k)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def norm_latex(x):
    """Light normalisation. It is deliberately conservative: two equivalent answers written
    differently are counted wrong. That under-counts accuracy uniformly across every arm, and since
    every comparison here is paired on the same problems, a uniform under-count cannot bias the
    DIFFERENCES the experiment reports."""
    if x is None:
        return None
    x = x.strip().strip("$").strip()
    for a, b in ((r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\,", ""), (r"\;", ""),
                 (r"\ ", ""), (r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"),
                 ("dfrac", "frac"), ("tfrac", "frac"), (" ", ""), (r"\%", ""), ("%", "")):
        x = x.replace(a, b)
    x = re.sub(r"\\text\{([^}]*)\}", r"\1", x)
    x = re.sub(r"\\mbox\{([^}]*)\}", r"\1", x)
    x = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", x)
    x = x.rstrip(".").rstrip()
    x = re.sub(r"^\\$|\\$$", "", x)
    return x


def gsm8k_gold(row):
    return row["answer"].split("####")[-1].strip().replace(",", "")


def gsm8k_pred(text):
    if "####" in text:
        m = NUM.search(text.split("####")[-1])
        if m:
            return m.group(0).replace(",", "").rstrip(".")
    m = NUM.findall(text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def num_same(a, b):
    try:
        return abs(float(str(a).replace(",", "")) - float(str(b).replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return False


def gsm8k_same(a, b):
    return False if a is None else (num_same(a, b) or a == b)


def math_same(a, b):
    a, b = norm_latex(a), norm_latex(b)
    return False if a is None else (a == b or num_same(a, b))


IFEVAL_INSTR = ""      # the IFEval prompt already carries its own constraints
MBPP_INSTR = ("Write the function in a single ```python code block. "
              "Define exactly the function the tests call, and output no other code.")


def mbpp_code(text):
    """The last fenced python block, else the whole reply."""
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[-1] if m else text


# No added instruction. The whole measurement is whether the model declines, so telling it to be
# helpful, or to be careful, would plant the disposition we are trying to observe drift in.
XSTEST_INSTR = ""

TASKS = {
    "gsm8k": dict(hf=("openai/gsm8k", "main", "test"), q="question", instr=GSM8K_INSTR,
                  gold=gsm8k_gold, pred=gsm8k_pred, same=gsm8k_same, max_new=400),
    "math": dict(hf=("HuggingFaceH4/MATH-500", None, "test"), q="problem", instr=MATH_INSTR,
                 gold=lambda r: r["answer"], pred=boxed, same=math_same, max_new=768),
    # graded: a response can satisfy some constraints and miss others, so pools are far less often
    # uniform, which is the limitation the MATH design runs into
    "ifeval": dict(hf=("google/IFEval", None, "train"), q="prompt", instr=IFEVAL_INSTR,
                   gold=lambda r: r, pred=lambda t: t, same=None, max_new=1024, graded=True),
    # Two-sided: comply on a safe prompt, refuse on an unsafe one. The safe and unsafe halves are
    # matched by surface form ("kill a Python process" against "kill a person"), so a grader that
    # drifts toward refusing anything that reads dangerous loses points on the safe half while a
    # one-sided safety score would call it an improvement. That dissociation is the point.
    "xstest": dict(hf=("Paul/XSTest", None, "train"), q="prompt", instr=XSTEST_INSTR,
                   gold=lambda r: dict(label=r["label"], type=r["type"], prompt=r["prompt"]),
                   pred=lambda t: t, same=None, max_new=512, graded=True),
    "mbpp": dict(hf=("google-research-datasets/mbpp", "full", "test"),
                 q=lambda r: (r["text"] + "\n\nYour code must pass these tests:\n"
                              + "\n".join(r["test_list"])),
                 instr=MBPP_INSTR, gold=lambda r: r, pred=mbpp_code, same=None, max_new=640),
}


def generate(model_id, questions, n, temperature, max_new, batch, instr):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    tick(f"loaded generator {model_id}")
    texts = []
    for q in questions:
        msg = [{"role": "user", "content": f"{q}\n\n{instr}"}]
        if getattr(tok, "chat_template", None):
            texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        else:
            texts.append(f"<|user|>\n{q}\n\n{instr}\n<|assistant|>\n")
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                 max_new_tokens=max_new, num_return_sequences=n,
                                 pad_token_id=tok.pad_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        out.extend(dec)
        tick(f"  generated {min(i+batch, len(texts))}/{len(texts)} prompts")
    del model
    torch.cuda.empty_cache()
    return out                                  # len = len(questions) * n, prompt-major


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--task", choices=sorted(TASKS), default="gsm8k")
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    T = TASKS[args.task]
    from datasets import load_dataset
    name, cfg, split = T["hf"]
    ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
    keep = np.arange(len(ds))
    if args.task == "ifeval":
        import verifiers as V
        keep = np.array([i for i in range(len(ds)) if V.ifeval_supported(ds[int(i)])])
        tick(f"IFEval: {len(keep)}/{len(ds)} prompts fully checkable, rest dropped")
    idx = keep[np.random.default_rng(0).permutation(len(keep))[: args.n_prompts]]
    qf = T["q"]
    qs = [qf(ds[int(i)]) if callable(qf) else ds[int(i)][qf] for i in idx]
    golds = [T["gold"](ds[int(i)]) for i in idx]
    mx = args.max_new or T["max_new"]
    tick(f"{len(qs)} {args.task} problems, {args.n_samples} samples each, max_new={mx}")

    resp = generate(args.gen, qs, args.n_samples, args.temperature, mx, args.batch, T["instr"])
    P, n = len(qs), args.n_samples
    assert len(resp) == P * n, f"expected {P*n} generations, got {len(resp)}"

    if args.task == "ifeval":
        import verifiers as V
        correct = np.array([[V.ifeval_fraction(golds[p], resp[p * n + j]) or 0.0
                             for j in range(n)] for p in range(P)], dtype=float)
    elif args.task == "xstest":
        import verifiers as V
        correct = np.array([[V.xstest_score(golds[p], resp[p * n + j])
                             for j in range(n)] for p in range(P)], dtype=float)
    elif args.task == "mbpp":
        import verifiers as V
        correct = np.array([[float(V.run_mbpp(T["pred"](resp[p * n + j]),
                                              list(golds[p]["test_list"]),
                                              [golds[p]["test_setup_code"]]
                                              if golds[p].get("test_setup_code") else []))
                             for j in range(n)] for p in range(P)], dtype=float)
    else:
        correct = np.array([[float(T["same"](T["pred"](resp[p * n + j]), golds[p]))
                             for j in range(n)] for p in range(P)], dtype=float)
    # a pool is dead when every candidate scores the same: the grader's pick cannot change the
    # outcome. For binary tasks this is exactly all-right or all-wrong; for graded ones it is
    # weaker, which is why IFEval should leave more pools live.
    live = correct.max(1) > correct.min(1)
    k = (correct >= 1.0).sum(1)
    tick(f"pools: all-max {(k == n).mean():.1%}, all-zero {(correct.max(1) == 0).mean():.1%}, "
         f"LIVE {live.mean():.1%}  <- live is the effective sample size")
    tick(f"mean per-sample score {correct.mean():.3f}, best-in-pool {correct.max(1).mean():.3f}")
    tick(f"verified: mean per-sample accuracy {correct.mean():.3f}, "
         f"at least one right on {correct.any(1).mean():.3f} of problems, "
         f"all right on {correct.all(1).mean():.3f}")

    import extract
    model, tok = extract.load()
    tick("loaded ArmoRM")
    phi = np.zeros((P, n, 19), dtype=np.float32)
    W = np.zeros((P, n, 19), dtype=np.float32)
    sc = np.zeros((P, n), dtype=np.float32)
    for p in range(P):
        r = extract.score_pairs(model, tok, [(qs[p], resp[p * n + j]) for j in range(n)])
        phi[p], W[p], sc[p] = r["phi"], r["w"], r["score"]
        if (p + 1) % 50 == 0:
            tick(f"  scored {p+1}/{P} problems")

    rng = np.random.default_rng(1)
    pick_rand = correct[np.arange(P), rng.integers(0, n, P)].mean()
    bon = correct[np.arange(P), sc.argmax(1)].mean()
    won = correct[np.arange(P), sc.argmin(1)].mean()
    orc = correct.max(1).mean()
    mixed = live.mean()

    print("\n" + "=" * 68)
    print(f"GENERATOR {args.gen}   TASK {args.task}   {P} problems x {n} samples")
    print(f"  random pick (= base accuracy)   {correct.mean():.3f}")
    print(f"  one random draw                 {pick_rand:.3f}")
    print(f"  ArmoRM worst-of-n               {won:.3f}")
    print(f"  ArmoRM best-of-n                {bon:.3f}")
    print(f"  oracle best-of-n (headroom)     {orc:.3f}")
    print(f"  live pools                      {mixed:.3f}   <- only these can show a lift")
    lift, room = bon - correct.mean(), orc - correct.mean()
    print(f"\n  LIFT  best-of-n minus random    {lift:+.3f}")
    print(f"  ROOM  oracle minus random       {room:+.3f}")
    print(f"  fraction of headroom captured   {lift/room if room > 0 else float('nan'):.3f}")
    print(f"  spread best minus worst         {bon-won:+.3f}   <- ranking signal, either sign")
    print("=" * 68)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Provenance. The scoring scripts already stamp these; generation did not, so the MATH and
    # XSTest pools carried no record of which GPU sampled them. fp32 scoring was shown to select
    # identically across H200 and A100, but generation reproducibility was never tested anywhere,
    # so at minimum the host has to be recoverable.
    np.savez_compressed(args.out, phi=phi, w=W, score=sc, correct=correct,
                        attributes=np.array(extract.ATTRIBUTES), gold=np.array(golds),
                        prompt_idx=idx[:P], generator=args.gen, task=args.task,
                        n_samples=n, temperature=args.temperature,
                        host=os.environ.get("SLURMD_NODENAME", "?"),
                        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"),
                        gen_dtype=str(next(iter([getattr(torch, "bfloat16")]), "")),
                        seed=int(args.seed) if hasattr(args, "seed") else -1)
    with open(args.out.replace(".npz", "_responses.json"), "w") as fh:
        json.dump({"questions": qs, "golds": golds, "responses": resp}, fh)
    tick(f"wrote {args.out}")


if __name__ == "__main__":
    main()
