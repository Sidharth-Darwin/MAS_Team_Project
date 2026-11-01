# Report experiments

There are exactly three independent report experiments:

1. `activation_comparison.py` — identity versus tanh.
2. `network_size_generalization.py` — train on N=3, test on N=3/50/100.
3. `plant_gain_transfer.py` — compare inference full-plant gain below/above the training gain.

`_common.py` contains shared ER graph generation, reproducibility, paired rollout,
metrics and plotting utilities. It does not implement or duplicate any plant,
controller, training or gain equations; those remain in `src/distributed_consensus`.

Each experiment has its own command and result directory. No command runs the
other two experiments.
