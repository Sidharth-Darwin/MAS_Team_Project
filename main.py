"""Command-line entry point for independent training and inference experiments."""
import sys

from experiments import compare_topologies, stress_test, train_controller, tune_hyperparameters

COMMANDS = {
    'train': train_controller,
    'topologies': compare_topologies,
    'stress': stress_test,
    'tune': tune_hyperparameters,
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h', '--help'):
        print('Usage: uv run python main.py {train|topologies|stress|tune} [options]')
        return 0
    if args[0] not in COMMANDS:
        raise SystemExit(f'Unknown command {args[0]!r}; choose from: {", ".join(COMMANDS)}')
    return COMMANDS[args[0]].main(args[1:])


if __name__ == '__main__':
    main()