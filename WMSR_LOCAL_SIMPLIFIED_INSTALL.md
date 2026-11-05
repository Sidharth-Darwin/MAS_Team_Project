# Simplified scalar W-MSR consensus demo update

This patch replaces the confusing coordinate-wise filtering GUI with one scalar W-MSR signal per agent, matching the supplied `wmsr_local(X, N, F=1)` algorithm.

## Apply

From the repository root:

```bash
unzip -o ~/Downloads/distributed_consensus_wmsr_local_simplified.zip -d .
```

## Verify

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
uv run --extra test python -m pytest tests simulation/tests -q
```

## Launch

```bash
uv run python -m simulation
```

## W-MSR interpretation

For each normal agent, sort its neighbours' scalar messages relative to its own value. Remove up to `F` smallest values below its own and up to `F` largest values above its own. Keep equals and other values, then average its own value with the retained neighbours.

With `F=1`, two different neighbours may be ignored in the same update: one low and one high. This is exactly the supplied pseudocode. `F=1` under the F-local assumption means each normal node has at most one malicious neighbour; it does not mean only one scalar value can be ignored.

The simulator sends one scalar controller signal per agent and uses the retained messages inside each learned controller's original correction law. The physical plant remains the original two-state model and continues to use `src/distributed_consensus/control/dynamics.py`.

The demo uses a malicious broadcast (the same false scalar value to every neighbour), not Byzantine equivocation (different false values per receiver). It has no actuator attack or jamming model. It is an abstract consensus-control test bench, not a practical robot simulation.
