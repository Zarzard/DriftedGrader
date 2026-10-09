"""ArmoRM extraction: our u = w·φ, read out of the real deployed LLM reward model.

RLHFlow/ArmoRM-Llama3-8B-v0.1 (Wang et al., Findings of EMNLP 2024, arXiv:2406.12845):
frozen FsfairX-LLaMA3-8B backbone -> linear regression head -> 19 interpretable objectives (φ),
prompt-conditional Mixture-of-Experts gating -> weights (w), score = w·φ.

  φ = output.rewards                                  # (N,19)  per (prompt,response)
  w = output.gating_output @ reward_transform_matrix.T# (N,19)  effective per-objective coeffs
  score = sum(φ * w)  == output.score                 # identity (verified in smoke)

Gating is conditioned on the PROMPT (hidden state at the assistant-header position), so w is
~shared across candidate responses to the same prompt = "which of the 19 criteria this grader
attends to for this prompt". That is exactly the attention-over-many-criteria object in §3.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

# order == regression-head column index (README `attributes`); verbatim spellings.
ATTRIBUTES = [
    "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
    "helpsteer-complexity", "helpsteer-verbosity",
    "ultrafeedback-overall_score", "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness", "ultrafeedback-honesty", "ultrafeedback-helpfulness",
    "beavertails-is_safe", "prometheus-score",
    "argilla-overall_quality", "argilla-judge_lm",
    "code-complexity", "code-style", "code-explanation",
    "code-instruction-following", "code-readability",
]
assert len(ATTRIBUTES) == 19


def load(device="cuda", dtype=torch.bfloat16):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, device_map=device, trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    return model, tok


@torch.no_grad()
def score_pairs(model, tok, pairs, device="cuda", max_length=4096):
    """pairs: list of (prompt, response). Returns dict with numpy arrays:
       phi (N,19) objective scores, w (N,19) effective gating coeffs, score (N,), gate_raw (N,19)."""
    T = model.reward_transform_matrix.data.float().cpu()          # (19,19)
    phis, ws, scores, gates = [], [], [], []
    for prompt, response in pairs:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
        ids = tok.apply_chat_template(messages, return_tensors="pt",
                                      truncation=True, max_length=max_length).to(device)
        out = model(ids)
        phi = out.rewards.float().cpu()                          # (1,19)
        gate = out.gating_output.float().cpu()                   # (1,19)
        w = gate @ T.T                                           # (1,19) effective per-objective coeffs
        phis.append(phi); ws.append(w); gates.append(gate)
        scores.append(out.score.float().cpu().reshape(-1))
    return {
        "phi": torch.cat(phis).numpy(),
        "w": torch.cat(ws).numpy(),
        "gate_raw": torch.cat(gates).numpy(),
        "score": torch.cat(scores).numpy(),
        "attributes": ATTRIBUTES,
    }
