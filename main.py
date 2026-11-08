"""Thin dispatcher for the four report experiments."""

import sys

from experiments import (
    activation_comparison,
    network_size_generalization,
    plant_gain_transfer,
    topology_generalization,
)

COMMANDS = {
    "activation": activation_comparison,
    "size": network_size_generalization,
    "gain": plant_gain_transfer,
    "topology": topology_generalization,
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("Usage: uv run python main.py {activation|size|gain|topology} [options]")
        return 0
    if args[0] not in COMMANDS:
        raise SystemExit(
            f"Unknown command {args[0]!r}; choose from: {', '.join(COMMANDS)}"
        )
    return COMMANDS[args[0]].main(args[1:])


if __name__ == "__main__":
    main()
