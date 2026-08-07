"""Tier-A simulation suite for grader drift in preference learning.

See ``md_files/tier_a_experiment_spec.md`` for the executable specification and
``md_files/grader_drift_identifiability_note.md`` for the theory it derives from.

Module map (spec component ids in comments):
    teacher.py     C1  teacher + population
    drift.py       C2  drift generator (event calendar + per-round processes)
    streams.py     C3  A0 item streams
    anchors.py     C4  anchor battery
    probe.py       C5  probing protocol
    diagnostics.py C6  operational (naive-detector) diagnostics
    estimator.py   C7  state-space (EKF/Laplace) drift estimator
    envs/grid.py   C8  A1 gridworld
    rm.py, policy.py  C8  reward model + policy

Support modules built in M0:
    _math.py     sigmoid / sigmoid_prime / logit
    seeding.py   master-seed -> named per-component Generators
    config.py    YAML config load + resolve
    logging_io.py  per-round JSONL logging + resolved-config copy
"""

__version__ = "0.1.0"
