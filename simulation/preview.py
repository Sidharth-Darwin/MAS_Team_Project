"""Headless numerical validation / demonstration (no GUI dependency)."""
from __future__ import annotations
import argparse
from .model import DualSimulation, DEFAULT_CHECKPOINT, POSITION_TOL, VELOCITY_TOL, SPREAD_TOL, HOLD_STEPS


def main(argv=None):
    p = argparse.ArgumentParser(description='Compare both modes using existing trained weights')
    p.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--agents', type=int, default=10)
    p.add_argument('--topology', default='checkpoint')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--pulse-at', type=int, default=-1,
                   help='Inject matched input pulse when k equals this value; -1 means no pulse')
    p.add_argument('--pulse-agent', type=int, default=3,
                   help='1-based agent number')
    p.add_argument('--pulse-strength', type=float, default=8.)
    p.add_argument('--csv', default='', help='Optional export location')
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error('--steps must be positive')
    sim = DualSimulation(args.checkpoint, args.agents, args.topology, args.seed)
    for _ in range(args.steps):
        if sim.k == args.pulse_at:
            sim.pulse(args.pulse_agent - 1, args.pulse_strength, 3)
        sim.step()
    print(f'Existing checkpoint: {args.checkpoint}')
    print(f'N={sim.n}, graph={args.topology}, seed={args.seed}, k={sim.k}; '
          f'W={sim.W.ravel().tolist()}, activation={sim.activation}')
    print(f'Readiness: max |q|<={POSITION_TOL}, max |v|<={VELOCITY_TOL}, '
          f'spread<={SPREAD_TOL}, maintained for {HOLD_STEPS} samples')
    for name in ('baseline', 'learned'):
        r = sim.latest(name)
        k = r['ready_step']
        status = ('not confirmed in horizon' if k is None else
                  f'first valid k={k} (since last intervention k={sim.segment_start})')
        print(f'{name:8} readiness={status}; error_hold_k={r["error_step"]}; '
              f'J_E={r["disagreement_cost"]:.6f}; J_U={r["energy"]:.6f}; '
              f'e_rms_final={r["error_rms"]:.4g}')
    if args.csv:
        print('Saved:', sim.export_csv(args.csv))


if __name__ == '__main__':
    main()
