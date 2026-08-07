"""Per-round JSONL logging and resolved-config copy.

Spec Section 1: "one JSONL row per round per run at
``runs/{exp}/{schedule}/{seed}/log.jsonl``, plus a copy of the resolved config".
The row schema below is the *superset*; unused fields are written as null so
that ``analysis/`` can read every experiment's logs with one schema and never
needs to touch anything except the JSONL.

Design rule (spec Section 1): analysis reads only these logs. So every quantity
a downstream analysis needs must appear here. The field list is the contract;
v0.2 amendments (dual-reference harm, effective drift magnitude, raw+standardized
op_temp_inv, realized rho, ...) extend it additively -- append to ROW_FIELDS,
never repurpose an existing name.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Superset row schema (spec v0.2 Section 1). Order is for human readability of the
# JSONL; readers key by name. Unused-in-this-experiment fields are emitted as null.
# Extend additively (append) -- never repurpose an existing name.
ROW_FIELDS: tuple[str, ...] = (
    "round",
    "seed",
    "schedule",
    "v_true",
    "m_k",              # filter posterior mean (C7); v_hat reserved for unit direction
    "cov_diag",
    "loglik",
    "anchor_p_hat",
    "D_dir",
    "D_norm",
    "D_crit",           # {contrast: signed logit change} (A1 contrastive panel)
    "rho_step_real",
    "rho_cum_real",
    "eff_drift_mag",    # ||P_ach (v_k - v_0)|| (A1)
    "op_agree",
    "op_flip",
    "op_temp_inv_std",  # primary (standardized margins)
    "op_temp_inv_raw",  # secondary (unstandardized; identified, flat under S0)
    "gold_wstar",       # frozen-reference harm J_{w*}(pi_k)
    "gold_wt",          # current-standard J_{w_t}(pi_k)
    "regret_wt",        # J_{w_t}(pi*_{w_t}) - J_{w_t}(pi_k)
    "proxy_return",
    "e_shift",
    "e_epi",
    "n_labels_cum",
    "event_flags",
)


def _to_jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays to JSON-native types; pass through the rest."""
    if value is None:
        return None
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


class RunLogger:
    """Append-only JSONL writer that enforces the superset row schema."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "log.jsonl"
        self._fh = open(self.path, "w")

    def log_row(self, **fields: Any) -> None:
        """Write one round's row. Unknown keys raise; missing keys become null."""
        unknown = set(fields) - set(ROW_FIELDS)
        if unknown:
            raise KeyError(f"unknown log fields {sorted(unknown)}; extend ROW_FIELDS first")
        row = {name: _to_jsonable(fields.get(name, None)) for name in ROW_FIELDS}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_resolved_config(run_dir: str | Path, resolved_cfg: dict[str, Any]) -> Path:
    """Serialize the resolved config next to the log so a run self-describes."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "resolved_config.json"
    with open(out, "w") as fh:
        json.dump(_to_jsonable(resolved_cfg), fh, indent=2, sort_keys=True)
    return out
