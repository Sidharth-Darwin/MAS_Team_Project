# Install the W-MSR adversary integration

Apply this patch from the root of the existing `distributed_consensus` repository. It replaces only the listed simulation/documentation files and adds the reusable W-MSR package and tests. It does **not** replace the plant, controller, training, or dynamics implementation under `src/distributed_consensus/control/` or `src/distributed_consensus/learning/`.

```bash
cd ~/Downloads/distributed_consensus
unzip -o ~/Downloads/distributed_consensus_wmsr_adversary_patch.zip -d .
```

## Verify

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra test python -m pytest \
  tests/test_wmsr.py simulation/tests/test_simulation.py -q

uv run --extra test python -m pytest -q --ignore=tests/test_report_experiments.py
```

The first command runs the W-MSR and simulation integration tests. The second runs the project tests while skipping the existing long-running report-experiment smoke tests. The existing math and learning code is not changed by this patch.

## Run a headless malicious-broadcast example

```bash
uv run python -m simulation.preview \
  --agents 10 \
  --topology complete \
  --adversary 1 \
  --wmsr-f 1 \
  --attack-mode malicious \
  --threat-model F-local-malicious \
  --steps 100 \
  --csv simulation/logs/wmsr_malicious.csv
```

## Run a headless Byzantine example with multiple adversaries

```bash
uv run python -m simulation.preview \
  --agents 10 \
  --topology complete \
  --adversary 1 3 \
  --wmsr-f 2 \
  --attack-mode byzantine \
  --threat-model F-total-byzantine \
  --steps 100 \
  --csv simulation/logs/wmsr_byzantine.csv
```

Use `--wmsr-f 0` to disable filtering while leaving the adversary active. This is useful for a matched attack-only comparison. It is not a W-MSR-resilient run.

## Launch the GUI

```bash
uv run python -m simulation
```

On the GUI's adversary row, enter `off`, `2`, or a comma-separated list such as `2,5`. Select `F`, the behavior (malicious broadcast or Byzantine equivocation), and the F-local/F-total threat model, then apply the configuration. The UI validates the adversary-count assumption before applying the scenario.

## What is and is not claimed

- `src/distributed_consensus/robust/wmsr.py` implements the canonical **scalar** W-MSR trimming/update and exact small-graph robustness checks.
- The existing simulator has two-state vectors. It applies the scalar filter independently to each coordinate, then feeds the filtered disagreement into the existing learned correction. This is an explicitly labeled **coordinate-wise vector adaptation**, not a general vector-W-MSR theorem.
- The compromised node's physical actuator command is separate from its forged message. W-MSR filters messages; it cannot physically stop a compromised actuator.
- The original fixed-Laplacian small-gain certificate does not automatically apply to this state-dependent filtering adaptation.
- Exact graph-robustness checking is limited to 14 nodes. Larger graphs are marked **not checked**, not assumed to satisfy a W-MSR condition.
- The GUI's baseline remains local-only and is not physically affected by neighbor-message filtering. W-MSR affects the learned controller's neighbor-error signal. To compare the learned branch with filtering on versus off, run the same scenario twice, changing only `W-MSR F`.

See `simulation/README.md` and `WMSR_IMPLEMENTATION_NOTES.md` for the detailed algorithm, topology assumptions, and limitations.
