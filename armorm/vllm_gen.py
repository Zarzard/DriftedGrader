"""Batch generation through vLLM, run as a subprocess from its own environment.

HuggingFace `generate()` measured 8.53 s per prompt for 8 samples, about 600 tokens/s aggregate on
an A6000 that should reach 1500-3000. It has no paged KV cache and no continuous batching, so most
of the compute goes on padding: every sequence in a batch pads to the longest, and MATH outputs
range from 50 to 640 tokens.

This lives in a separate env because the shared one is python 3.8, which caps vLLM at a release
pinned to a different torch than everything else depends on. Input and output are JSON on disk, so
the two environments never share a process.

Usage: python vllm_gen.py <spec.json> <out.json>
  spec: {model, adapter|null, prompts[], n, temperature, max_new}
  out:  {responses[]}   prompt-major, n consecutive samples per prompt
"""
import json
import sys


def main(spec_path, out_path):
    spec = json.load(open(spec_path))
    from vllm import LLM, SamplingParams
    # gpu_memory_utilization must be derived from what is ACTUALLY free, not hard-coded. vLLM
    # interprets it as a fraction of TOTAL device memory and refuses to start if that exceeds free
    # memory. The parent stage1 process keeps a CUDA context and ~13 GiB of cached allocator blocks
    # alive across the HF training step, so a fixed 0.90 killed two 2-hour runs with:
    #   "Free memory on device cuda:0 (31.19/44.42 GiB) on startup is less than desired GPU memory
    #    utilization (0.9, 39.98 GiB)"
    # Leave 8% of free memory as headroom for the engine's own allocations outside the KV pool.
    import torch
    free_b, total_b = torch.cuda.mem_get_info()
    util = max(0.25, min(0.90, (free_b * 0.92) / total_b))
    print(f"vllm_gen: {free_b/2**30:.1f} of {total_b/2**30:.1f} GiB free "
          f"-> gpu_memory_utilization={util:.3f}", flush=True)
    kw = dict(model=spec["model"], dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=2048, enable_prefix_caching=True)
    if spec.get("adapter"):
        kw.update(enable_lora=True, max_lora_rank=32)
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in spec["prompts"]]
    sp = SamplingParams(n=spec["n"], temperature=spec["temperature"],
                        top_p=0.95 if spec["temperature"] > 0 else 1.0,
                        max_tokens=spec["max_new"], seed=spec.get("seed"))
    gen_kw = {}
    if spec.get("adapter"):
        from vllm.lora.request import LoRARequest
        gen_kw["lora_request"] = LoRARequest("adapter", 1, spec["adapter"])
    outs = llm.generate(texts, sp, **gen_kw)
    # vLLM returns results in input order; flatten prompt-major to match the HF path exactly
    resp = []
    for o in outs:
        cands = [c.text for c in o.outputs]
        while len(cands) < spec["n"]:          # defensive: never return a short pool
            cands.append(cands[-1] if cands else "")
        resp.extend(cands[: spec["n"]])
    assert len(resp) == len(spec["prompts"]) * spec["n"], \
        f"{len(resp)} responses for {len(spec['prompts'])}x{spec['n']}"
    json.dump({"responses": resp}, open(out_path, "w"))
    print(f"vllm_gen: {len(spec['prompts'])} prompts x {spec['n']} -> {len(resp)} responses")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
