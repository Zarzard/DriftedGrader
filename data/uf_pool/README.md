# UltraFeedback candidate pool

The prompt pool the closed-loop UltraFeedback arms run on. `uf_gold.load_pool()` reads both files;
they must sit together and keep these names, since the loader derives the JSON path from the NPZ
path by substitution.

- `uf_armorm.npz` -- per-candidate ArmoRM criterion scores and `raw_gold`, the UltraFeedback GPT-4
  ratings used as the gold label.
- `uf_armorm_responses.json` -- the prompts and the candidate responses themselves.

Point `POOL` in `armorm/uf_gold.py` at this directory. Committed here rather than copied by hand
because the experiment cannot run without it and an out-of-band file transfer is a step that goes
wrong. 11 MB, well inside GitHub's limits.
