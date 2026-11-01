# Install the four independent experiment scripts

Extract this ZIP **into the existing project directory**. It contains changed Python scripts/tests and `results/README.md` only; it does not contain or replace the mathematical library in `src/`.

```bash
cd ~/Downloads/distributed_consensus
unzip -o ~/Downloads/distributed_consensus_four_experiments_REPLACEMENT.zip -d .
```

Remove the following *obsolete experiment runners and their tests*, if present. This does **not** delete the reusable source package or its mathematical tests:

```bash
rm -f experiments/compare_topologies.py experiments/mixed_gain_transfer.py \
  experiments/report_studies.py experiments/run_report_suite.py \
  experiments/stress_test.py experiments/train_controller.py \
  experiments/tune_hyperparameters.py \
  tests/test_report_studies.py tests/test_report_suite.py
```

Then verify and run small checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra test python verify_math.py
uv run python experiments/activation_comparison.py --quick --out results/smoke_01
uv run python experiments/network_size_generalization.py --quick --out results/smoke_02
uv run python experiments/plant_gain_transfer.py --quick --out results/smoke_03
uv run python experiments/topology_generalization.py --quick --out results/smoke_04
```

Full run commands are in `results/README.md`. These smoke runs generate real per-agent/state/control PNGs but are not statistically meaningful report results. Remove `results/smoke_*` when satisfied.
