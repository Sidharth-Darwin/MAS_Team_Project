# W-MSR GUI layout and metric-label update

This update replaces only:

- `simulation/gui.py`
- `simulation/README.md`

It does not change the W-MSR algorithm, controller equations, physical dynamics, checkpoints, or tests.

From the repository root, apply it with:

```bash
unzip -o ~/Downloads/wmsr_gui_layout_metrics_update.zip -d .
uv run python -m simulation
```

Keyboard zoom:

- `Ctrl + +` or `Ctrl + =`: zoom in
- `Ctrl + -`: zoom out
- `Ctrl + 0`: reset to 100%

The interface zoom range is 90%–115% so the table and plot remain inside the available window.
