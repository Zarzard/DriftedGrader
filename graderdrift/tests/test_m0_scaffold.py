"""M0 plumbing tests: RNG spawning, config resolve, and a valid dummy run.

Definition of done (spec Section 7, M0): pytest green; one dummy run writes
valid JSONL.
"""
import json

import numpy as np

from graderdrift.config import resolve_config
from graderdrift.experiments._dummy import run_dummy
from graderdrift.logging_io import ROW_FIELDS
from graderdrift.seeding import COMPONENTS, spawn_generators


def test_seed_spawning_reproducible_and_independent():
    a = spawn_generators(42)
    b = spawn_generators(42)
    # Same master seed -> identical per-component streams.
    for name in COMPONENTS:
        xa = a[name].standard_normal(20)
        xb = b[name].standard_normal(20)
        assert np.array_equal(xa, xb)
    # Different components -> different streams (independence, not identity).
    c = spawn_generators(42)
    assert not np.array_equal(
        c["teacher"].standard_normal(20), c["stream"].standard_normal(20)
    )
    # Different master seed -> different streams.
    d = spawn_generators(43)
    assert not np.array_equal(
        spawn_generators(42)["teacher"].standard_normal(20),
        d["teacher"].standard_normal(20),
    )


def test_resolve_config_stamps_seed_and_is_a_copy():
    cfg = {"exp": "x", "nested": {"a": 1}}
    resolved = resolve_config(cfg, seed=7)
    assert resolved["seed"] == 7
    resolved["nested"]["a"] = 999
    assert cfg["nested"]["a"] == 1  # deep copy, original untouched


def test_dummy_run_writes_valid_jsonl(tmp_path):
    run_dir = tmp_path / "dummy" / "S0" / "0"
    log_path = run_dummy(run_dir, seed=0, n_rounds=3, d=4)
    assert log_path.exists()
    assert (run_dir / "resolved_config.json").exists()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        row = json.loads(line)  # must parse
        assert set(row.keys()) == set(ROW_FIELDS)  # exact superset schema
        assert row["round"] == i
        assert row["schedule"] == "S0"
        assert row["v_true"] is not None and len(row["v_true"]) == 4
        # Unused fields are present and null, not missing.
        assert "gold_wstar" in row and row["gold_wstar"] is None

    # resolved config round-trips as JSON and carries the seed.
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    assert resolved["seed"] == 0
