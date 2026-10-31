
#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   bash check_everything.sh
#   bash check_everything.sh --full
#
# Default: short, representative end-to-end smoke tests.
# --full: additionally evaluate 100- and 1000-agent networks.

FULL=0
if [[ "${1:-}" == "--full" ]]; then
    FULL=1
fi

cd "$(dirname "$0")"

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT="results/verification_${RUN_ID}"
mkdir -p "$OUT"

PASS=0
FAIL=0
SKIP=0

run_check() {
    local name="$1"
    shift

    echo
    echo "================================================"
    echo "CHECK: $name"
    echo "================================================"

    if "$@" 2>&1 | tee "$OUT/${name}.log"; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

skip_check() {
    echo "[SKIP] $1"
    SKIP=$((SKIP + 1))
}

echo "Verification results: $OUT"

# 1. Installation and Python environment
run_check environment uv run python -c '
import sys
import numpy, scipy, torch, matplotlib
import distributed_consensus

print("Python:", sys.version)
print("Package:", distributed_consensus.__file__)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
'

# 2. Import all installed project modules
run_check module_imports uv run python -c '
from distributed_consensus.control import plants
from distributed_consensus.control import controllers
from distributed_consensus.control import dynamics
from distributed_consensus.control import plant_gain
from distributed_consensus import graphs
from distributed_consensus.learning import training
from distributed_consensus.evaluation import metrics
from distributed_consensus.evaluation import reporting
print("All eight research modules imported successfully.")
'

# 3. Compile all Python source files
run_check syntax uv run python -m compileall -q \
    src main.py config.py verify_math.py experiments tests

# 4. Mathematical and regression tests
run_check pytest env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    uv run --extra test python verify_math.py

# 5. Verify all executable CLI entry points
run_check main_cli uv run python main.py --help

run_check topology_cli uv run python \
    experiments/compare_topologies.py --help

run_check tuner_cli uv run python \
    experiments/tune_hyperparameters.py --help

run_check stress_cli uv run python \
    experiments/stress_test.py --help

# 6. Small, fresh end-to-end training run
run_check training_pipeline uv run python main.py \
    --suite quick \
    --run-name "verification_${RUN_ID}" \
    --results-root "$OUT/main" \
    --epochs 5 \
    --horizon 12 \
    --batch 8 \
    --eval-samples 4 \
    --lambda-values 0 \
    --inference-sizes 10 30 \
    --device cpu

# 7. Find an actual checkpoint (never use a placeholder)
CHECKPOINT=$(find "$OUT/main" results \
    -type f \
    \( -name "best_validation_checkpoint.npz" \
       -o -name "checkpoint.npz" \) \
    -print 2>/dev/null | sort | head -n 1)

if [[ -n "$CHECKPOINT" ]]; then
    echo "Using checkpoint: $CHECKPOINT"

    # 8. Validate saved checkpoint contents
    run_check checkpoint_load uv run python -c '
import sys
import numpy as np
from distributed_consensus.evaluation.reporting import load_checkpoint

path = sys.argv[1]
system, graph, W, activation, meta = load_checkpoint(path)
print("Checkpoint:", path)
print("Weight shape:", np.asarray(W).shape)
print("Weight norm:", np.linalg.norm(W, 2))
print("Activation:", activation)
print("Loaded successfully.")
' "$CHECKPOINT"

    # 9. Six graph topologies
    run_check topology_experiment uv run python \
        experiments/compare_topologies.py \
        --checkpoint "$CHECKPOINT" \
        --sizes 3 5 10 \
        --samples 2 \
        --horizon 25 \
        --out "$OUT/topologies"

    # 10. Disturbance and generalization stress tests
    run_check stress_experiment uv run python \
        experiments/stress_test.py \
        --checkpoint "$CHECKPOINT" \
        --sizes 10 30 \
        --samples 2 \
        --horizon 25 \
        --out "$OUT/stress"

    if [[ "$FULL" == 1 ]]; then
        # Large star/ring and other supported sparse graphs
        run_check large_topologies uv run python \
            experiments/compare_topologies.py \
            --checkpoint "$CHECKPOINT" \
            --sizes 100 1000 \
            --samples 1 \
            --horizon 25 \
            --out "$OUT/large_topologies"

        run_check large_stress uv run python \
            experiments/stress_test.py \
            --checkpoint "$CHECKPOINT" \
            --sizes 100 1000 \
            --samples 1 \
            --horizon 25 \
            --out "$OUT/large_stress"
    fi
else
    echo "No checkpoint found after training."
    skip_check checkpoint_load
    skip_check topology_experiment
    skip_check stress_experiment
fi

# 11. Independent hyperparameter tuner smoke test
run_check hyperparameter_tuner uv run python \
    experiments/tune_hyperparameters.py \
    --out "$OUT/tuning" \
    --search grid \
    --learning-rates 0.002 \
    --batch-sizes 8 \
    --epochs 5 \
    --horizons 12 \
    --activations identity \
    --lambda-values 0 0.1 \
    --seeds 7 \
    --validation-samples 8 \
    --selection-horizon 12 \
    --max-candidates 2 \
    --refine-top 0 \
    --finalists 0 \
    --device cpu

# 12. Check tuner resume behavior (no repeated training)
run_check tuner_resume uv run python \
    experiments/tune_hyperparameters.py \
    --out "$OUT/tuning" \
    --search grid \
    --learning-rates 0.002 \
    --batch-sizes 8 \
    --epochs 5 \
    --horizons 12 \
    --activations identity \
    --lambda-values 0 0.1 \
    --seeds 7 \
    --validation-samples 8 \
    --selection-horizon 12 \
    --max-candidates 2 \
    --refine-top 0 \
    --finalists 0 \
    --device cpu

# Final summary
echo
echo "================================================"
echo "FINAL VERIFICATION SUMMARY"
echo "================================================"
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Skipped: $SKIP"
echo "Results: $OUT"
echo
echo "Look for [FAIL] above and inspect the relevant .log."
echo "The resume test also needs manual log inspection"
echo "to confirm completed candidates were skipped."

if [[ "$FAIL" -eq 0 ]]; then
    echo "ALL EXECUTED COMMANDS PASSED"
    exit 0
else
    echo "SOME COMMANDS FAILED"
    exit 1
fi
