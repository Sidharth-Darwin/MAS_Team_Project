# Install the three independent report experiments

This add-on is designed for the existing `distributed_consensus` repository.
It does **not** replace or modify the mathematical implementation under
`src/distributed_consensus/`.

## Files added or replaced

### Added

- `experiments/_common.py` — shared ER graph generation, paired rollout metrics,
  gain calibration, reproducibility snapshots and plotting.
- `experiments/activation_comparison.py` — Experiment 01.
- `experiments/network_size_generalization.py` — Experiment 02.
- `experiments/plant_gain_transfer.py` — Experiment 03.
- `experiments/README.md` — experiment-script guide.
- `results/README.md` — result interpretation and exact reproduction commands.
- `tests/test_report_experiments.py` — mathematical and smoke tests for the three studies.

### Replaced

- `main.py` — thin dispatcher for one selected experiment; it does not run all three.
- `tests/test_cli_and_simulation.py` — retains the external simulator stepping test.
- `tests/test_dispatch_and_graph_files.py` — retains graph-file tests and tests the new dispatcher.
- `tests/test_incremental.py` — retains core graph, metric, training, checkpoint and gain tests;
  removes tests that imported the retired topology-comparison/tuner scripts.

### Unchanged

All files under `src/distributed_consensus/` other than the obsolete `config.py`
listed below. In particular, plants, exact pseudoinverse homogenization, graph
Laplacian, controller, physical dynamics, training, metrics and gain calculations
are not changed.

## Old files to delete after installing and verifying the replacements

Delete these retired experiment scripts:

```text
experiments/run_report_suite.py
experiments/report_studies.py
experiments/mixed_gain_transfer.py
experiments/train_controller.py
experiments/compare_topologies.py
experiments/stress_test.py
experiments/tune_hyperparameters.py
```

Delete these tests for the retired experiment runners:

```text
tests/test_report_suite.py
tests/test_report_studies.py
```

Delete this unused configuration module. It was only imported by the retired
`experiments/train_controller.py` script:

```text
src/distributed_consensus/config.py
```

Do **not** delete the core source modules, `verify_math.py`, `pyproject.toml`,
`uv.lock`, `tests/test_math.py`, `tests/test_training_fixes.py`, or
`tests/test_evaluation_fixes.py`.

## Installation and verification

Extract this ZIP into the repository root, allowing it to replace the listed
files. Then run:

```bash
cd ~/Downloads/distributed_consensus
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra test python verify_math.py
```

Expected local verification for the supplied source snapshot: **76 passed,
1 skipped** (the CUDA-only consistency test is skipped when CUDA is unavailable).

Only after the new tests pass, delete the retired files listed above. Run the
verification command again. The result README is explicitly unignored by the
included `.gitignore`; generated result subdirectories remain ignored.

## Run the experiments independently

```bash
uv run python experiments/activation_comparison.py --help
uv run python experiments/network_size_generalization.py --help
uv run python experiments/plant_gain_transfer.py --help
```

A small pipeline smoke test:

```bash
uv run python experiments/activation_comparison.py --quick --out results/smoke_activation
uv run python experiments/network_size_generalization.py --quick --out results/smoke_size
uv run python experiments/plant_gain_transfer.py --quick --out results/smoke_gain
```

The `--quick` preset tests only N=3 and is not report evidence. The report-scale
commands and metric definitions are in `results/README.md`.
