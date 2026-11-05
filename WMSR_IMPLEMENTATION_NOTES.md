# W-MSR compatibility: implementation and mathematical scope

## Files changed / added

- `src/distributed_consensus/robust/wmsr.py` — canonical scalar W-MSR update,
  coordinate-wise vector filtering adapter, exact small-graph robustness tests,
  and adversary-count validation.
- `src/distributed_consensus/robust/__init__.py` — public exports.
- `simulation/model.py` — routes actual per-agent messages, simulates malicious
  broadcast or Byzantine equivocation, filters received messages before the
  existing learned correction, and reports normal-agent metrics. The GUI and headless CLI accept multiple compromised agent IDs and validate F-local/F-total counts.
- `simulation/gui.py` — adversary/W-MSR controls, adversary highlighting,
  normal-agent disagreement chart label, and topology-condition status.
- `simulation/README.md` — assumptions, limitations, graph conditions, and
  usage guidance.
- `tests/test_wmsr.py` and `simulation/tests/test_simulation.py` — W-MSR and
  adversarial integration tests.

No changes were made to the original plant/controller/training implementations
in `src/distributed_consensus/control/` or `src/distributed_consensus/learning/`.
The simulator imports and uses those existing APIs as before.

## Mathematical interpretation

Canonical W-MSR is a scalar neighbor-message filtering and convex-update
algorithm. This project has a two-state physical agent and a feedback controller
whose correction depends on neighbor disagreement. The simulator implements
coordinate-wise scalar W-MSR trimming as a front-end filter, then feeds the
filtered two-dimensional disagreement into the original controller. That is a
reasonable robustness-oriented adaptation, but it is not canonical vector
W-MSR and does not inherit its resilient-consensus theorem automatically.

The original small-gain condition was derived for a fixed Laplacian feedback
operator. W-MSR selection depends on current messages, so it makes the effective
feedback map state-dependent and potentially switching. The old small-gain
certificate is not asserted for the filtered system.

Graph robustness checking is exact only up to 14 nodes due to exponential
complexity. Above that size, the report is explicitly unknown/not checked.
