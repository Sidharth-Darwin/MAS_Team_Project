"""Readable Tkinter GUI for the scalar local W-MSR consensus test bench."""
from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import font as tkfont

import numpy as np

from .model import (
    DEFAULT_BENCHMARK_CHECKPOINT, METHODS, METHOD_LABELS, METHOD_COLORS,
    TOPOLOGIES, MultiPolicySimulation,
)

BG = '#0b1220'
PANEL = '#142236'
PANEL_ALT = '#1b2b42'
STROKE = '#2c405b'
WHITE = '#edf4fc'
MUTED = '#aebfd2'
RED = '#f08088'
ORANGE = '#f0b45d'
TEAL = '#43cbb0'
FONT = 'DejaVu Sans'


class DemoApp:
    def __init__(self, root, benchmark=DEFAULT_BENCHMARK_CHECKPOINT, seed=3):
        self.root = root
        self.benchmark = Path(benchmark)
        self.seed = int(seed)
        self.zoom = 1.0
        self._fonts = {}
        self.sim = MultiPolicySimulation(
            self.benchmark, topology='complete', seed=self.seed,
            adversary_ids=(9,), F=1,
        )
        self.selected_method = 'paper_gnn_wmsr'
        self.selected_agent = 0
        self.running = False
        self.rate = 12
        self.last_time = time.perf_counter()
        self.accumulator = 0.0
        self.node_hitboxes = []
        self.method_hitboxes = []
        self._build_ui()
        self._update_robustness_status(show_warning=False)
        self._refresh()
        self.root.after(40, self._tick)

    def _font(self, size, weight='normal'):
        """Return a named Tk font so Ctrl+plus can resize the whole interface."""
        key = (int(size), weight)
        if key not in self._fonts:
            self._fonts[key] = tkfont.Font(
                root=self.root, family=FONT,
                size=max(8, round(size * self.zoom)), weight=weight,
            )
        return self._fonts[key]

    def _set_zoom(self, value):
        self.zoom = min(1.15, max(0.90, float(value)))
        for (base_size, weight), font in self._fonts.items():
            font.configure(size=max(8, round(base_size * self.zoom)))
        self._update_footer()
        self.root.update_idletasks()
        self._refresh()
        return 'break'

    def _zoom_in(self, _event=None):
        return self._set_zoom(self.zoom * 1.1)

    def _zoom_out(self, _event=None):
        return self._set_zoom(self.zoom / 1.1)

    def _zoom_reset(self, _event=None):
        return self._set_zoom(1.0)

    def _update_footer(self):
        if hasattr(self, 'footer_label'):
            percent = round(self.zoom * 100)
            self.footer_label.configure(
                text=(
                    'Click a controller or agent for details.  '
                    f'Ctrl + + / Ctrl + −: zoom  ·  Ctrl + 0: reset ({percent}%).'
                )
            )

    def _button(self, parent, label, command, primary=False):
        return tk.Button(
            parent, text=label, command=command,
            font=self._font(14, 'bold'), bg=TEAL if primary else PANEL_ALT,
            fg=BG if primary else WHITE, activebackground='#3d5976',
            activeforeground=WHITE, relief='flat', bd=0, padx=15, pady=8,
            cursor='hand2', highlightthickness=0,
        )

    def _combo(self, parent, variable, values, width=12):
        return ttk.Combobox(parent, textvariable=variable, values=values,
                            width=width, state='readonly', font=self._font(14))

    def _label(self, parent, text):
        return tk.Label(parent, text=text, bg=BG, fg=MUTED,
                        font=self._font(14, 'bold'))

    def _entry(self, parent, variable, width=8):
        return tk.Entry(parent, textvariable=variable, width=width, bg=PANEL_ALT,
                        fg=WHITE, insertbackground=WHITE, relief='flat',
                        font=self._font(14), justify='center')

    def _build_ui(self):
        self.root.title('W-MSR Consensus Test Bench')
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        width = min(1600, max(1024, sw - 24))
        height = min(1100, max(700, sh - 70))
        self.root.geometry(f'{width}x{height}+8+8')
        self.root.minsize(980, 700)
        self.root.configure(bg=BG)

        header = tk.Frame(self.root, bg=BG, padx=18, pady=8)
        header.pack(fill='x')
        tk.Label(header, text='W-MSR consensus test bench', bg=BG,
                 fg=WHITE, font=self._font(25, 'bold')).pack(anchor='w', pady=(2, 0))
        tk.Label(
            header,
            text='Five matched controllers · one scalar signal per agent · one malicious broadcaster',
            bg=BG, fg=MUTED, font=self._font(13),
        ).pack(anchor='w', pady=(2, 0))

        toolbar = tk.Frame(self.root, bg=BG, padx=18, pady=5)
        toolbar.pack(fill='x')
        self.play_button = self._button(toolbar, '▶ PLAY', self.play_pause, True)
        self.play_button.pack(side='left', padx=(0, 6))
        self._button(toolbar, 'STEP', self.advance).pack(side='left', padx=4)
        self._button(toolbar, 'RESTART', self.restart).pack(side='left', padx=4)
        self._button(toolbar, 'SAVE FULL RUN', self.save_run).pack(side='left', padx=4)
        self.step_label = tk.Label(toolbar, text='Step 0 / 250', bg=BG, fg=WHITE,
                                   font=self._font(14, 'bold'))
        self.step_label.pack(side='left', padx=(16, 10))
        self._button(toolbar, 'HELP', self.show_help).pack(side='left', padx=4)

        config = tk.Frame(self.root, bg=BG, padx=18, pady=5)
        config.pack(fill='x')
        self._label(config, 'Graph').pack(side='left', padx=(0, 5))
        self.topology_var = tk.StringVar(value='complete')
        self._combo(config, self.topology_var, TOPOLOGIES, 12).pack(side='left', padx=(0, 18))
        self._label(config, 'Malicious agent ID(s)').pack(side='left', padx=(0, 5))
        self.adversary_var = tk.StringVar(value='10')
        self._entry(config, self.adversary_var, 9).pack(side='left', padx=(0, 18))
        self._label(config, 'F').pack(side='left', padx=(0, 5))
        self.f_var = tk.StringVar(value='1')
        self._combo(config, self.f_var, ('1', '2'), 4).pack(side='left', padx=(0, 14))
        self._button(config, 'APPLY', self.apply_config, True).pack(side='left', padx=4)
        note_row = tk.Frame(self.root, bg=BG, padx=18, pady=3)
        note_row.pack(fill='x')
        self.config_note = tk.Label(
            note_row,
            text='F = max malicious neighbours per normal agent. W-MSR ignores extreme neighbour values.',
            bg=BG, fg=MUTED, font=self._font(13), anchor='w', justify='left', wraplength=1200,
        )
        self.config_note.pack(side='left', fill='x', expand=True)

        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill='both', expand=True, padx=12, pady=(3, 6))
        self.canvas.bind('<Configure>', lambda _event: self._refresh())
        self.canvas.bind('<Button-1>', self._mouse_down)

        footer = tk.Frame(self.root, bg=BG, padx=18, pady=5)
        footer.pack(fill='x')
        self.footer_label = tk.Label(
            footer, text='Click a controller or agent for details.  Ctrl + + / Ctrl + −: zoom  ·  Ctrl + 0: reset (100%).',
            bg=BG, fg=MUTED, font=self._font(13), anchor='w', justify='left', wraplength=1100,
        )
        self.footer_label.pack(side='left', fill='x', expand=True)
        self._update_footer()

        self.root.bind('<space>', lambda _event: self.play_pause())
        self.root.bind('<Right>', lambda _event: self.advance())
        self.root.bind('<Escape>', lambda _event: self.pause())
        self.root.bind('<Control-r>', lambda _event: self.restart())
        for sequence in ('<Control-plus>', '<Control-equal>', '<Control-KP_Add>'):
            self.root.bind_all(sequence, self._zoom_in, add='+')
        for sequence in ('<Control-minus>', '<Control-KP_Subtract>'):
            self.root.bind_all(sequence, self._zoom_out, add='+')
        self.root.bind_all('<Control-0>', self._zoom_reset, add='+')

    def show_help(self):
        win = tk.Toplevel(self.root)
        win.title('How this demo works')
        win.geometry('980x800')
        win.minsize(820, 650)
        win.configure(bg=BG)
        outer = tk.Frame(win, bg=BG, padx=16, pady=14)
        outer.pack(fill='both', expand=True)
        text = tk.Text(
            outer, wrap='word', bg=PANEL, fg=WHITE, insertbackground=WHITE,
            relief='flat', padx=18, pady=16, font=self._font(16),
            spacing1=5, spacing3=7,
        )
        scroll = ttk.Scrollbar(outer, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        text.pack(side='left', fill='both', expand=True)
        explanation = """WHAT THIS DEMO IS

This is an abstract consensus-control test bench. It is not a robot, vehicle, or formation simulation. It shows how five controllers behave when one agent sends a false communication value.

THE FIVE CONTROLLERS

• Local-only: uses no neighbour information. It is a reference.
• Paper GNN: trained paper controller, using unfiltered messages.
• Paper GNN + W-MSR: the same trained controller, with W-MSR filtering.
• Small-gain: your trained controller, using unfiltered messages.
• Small-gain + W-MSR: the same trained controller, with W-MSR filtering.

Select a controller row to highlight it. Select a node to inspect its received messages.

ONE SCALAR MESSAGE

Each agent sends one scalar coordination signal. For the paper GNN it is H x_i. For the small-gain controller it is W^T x_i. W-MSR filters the neighbour signals before the controller forms its correction. The physical plant still has its original two states and is updated by the existing dynamics code.

THE W-MSR RULE YOU PROVIDED

For each normal receiver i:
1. Compare each neighbour's scalar value with i's own value.
2. From values below i, remove the F smallest.
3. From values above i, remove the F largest.
4. Keep values equal to i and all other neighbour values.
5. Average i's own value together with the retained neighbour values.

Example: own value 5; neighbour values [1, 4, 5, 7, 100]; F=1. Remove 1 and 100. Keep 4, 5 and 7. The new value is mean([5, 4, 5, 7]) = 5.25.

With F=1, at most two neighbour values can be ignored in one update: the smallest value below the receiver and the largest value above it. They may come from two different agents. This is exactly what the supplied pseudocode does. It does not mean two malicious neighbours are allowed, and no graph links are deleted.

WHAT F MEANS HERE

We use the F-local assumption: each normal agent has at most F malicious neighbours. For F=1, every normal agent may have no more than one malicious neighbour. The filter does not know which sender is malicious, so an honest neighbour with an extreme value can also be ignored.

THE ATTACK IN THIS DEMO

The red agent sends the same false scalar value to all of its neighbours. This is called a malicious broadcast.

Byzantine equivocation is different: the compromised agent sends different false values to different neighbours. This demo does NOT model equivocation, actuator attacks, or communication jamming.

GRAPH WARNING

For this F-local model, 3-robustness is a sufficient graph condition when F=1. A graph being connected is not enough. The graph panel reports whether this condition is met. If it is not met, the simulation can still run, but the usual scalar W-MSR guarantee cannot be claimed.

THE NUMBERS

• Step: current discrete-time step, out of 250.
• e_RMS: current mean-centred RMS disagreement among normal agents. This is the curve shown below.
• J_E: cumulative squared disagreement cost, summed over all steps so far.
• J_U: cumulative squared physical control-input energy, summed over all steps so far.

J_E and J_U are accumulated costs, not pieces that are added together to produce the current error. They cannot decrease; they flatten when each new step adds very little. The current e_RMS can keep decreasing while those accumulated totals barely change.

METRIC DEFINITIONS
For N normal agents and their two-dimensional states x_i(k):
  e_RMS(k) = sqrt((1/N) sum_i ||x_i(k) - mean_j x_j(k)||^2)
  J_E(k) = sum_{t=0}^{k-1} sum_i ||x_i(t) - mean_j x_j(t)||^2
  J_U(k) = sum_{t=0}^{k-1} sum_i u_i(t)^2
The GUI shows J_E and J_U separately; it does not combine them into one weighted score.

• Orange dashed arrow: the selected receiver ignored that sender's scalar value. The communication link itself is still present.

IMPORTANT LIMITATION

This demo is a consensus chart with a communication-network view and message audit. It is useful for inspecting the algorithm, but it is not a practical application simulation. W-MSR is applied to one scalar controller signal per agent. The classical scalar W-MSR theorem does not, by itself, prove stability of the complete two-state physical feedback system.

ZOOM
Ctrl + + (or Ctrl + =) zooms in. Ctrl + - zooms out. Ctrl + 0 resets to 100%.
"""
        text.insert('1.0', explanation)
        text.configure(state='disabled')

    def play_pause(self):
        self.running = not self.running
        self.play_button.configure(text='Ⅱ PAUSE' if self.running else '▶ PLAY')
        self.last_time = time.perf_counter()
        self._refresh()

    def pause(self):
        self.running = False
        self.play_button.configure(text='▶ PLAY')
        self._refresh()

    def advance(self):
        if self.sim.k < 250:
            self.sim.step()
        if self.sim.k >= 250:
            self.pause()
        self._refresh()

    def restart(self):
        self.pause()
        self.sim.reset()
        self._refresh()

    def apply_config(self):
        try:
            tokens = [part.strip() for part in self.adversary_var.get().split(',') if part.strip()]
            if tokens in (['0'], ['off']):
                adversaries = ()
            else:
                numbers = [int(value) for value in tokens]
                if not numbers or len(set(numbers)) != len(numbers):
                    raise ValueError('Enter unique agent IDs, or 0 to disable the adversary.')
                if any(value < 1 or value > self.sim.n for value in numbers):
                    raise ValueError(f'Agent IDs must be in [1, {self.sim.n}].')
                adversaries = tuple(value - 1 for value in numbers)
            candidate = MultiPolicySimulation(
                self.benchmark, topology=self.topology_var.get(), seed=self.seed,
                adversary_ids=adversaries, F=int(self.f_var.get()),
            )
            self.sim = candidate
            self.selected_agent = min(self.selected_agent, self.sim.n - 1)
            self.pause()
            self._update_robustness_status()
            self._refresh()
        except Exception as exc:
            messagebox.showerror('Invalid configuration', str(exc))

    def _update_robustness_status(self, show_warning=True):
        report = self.sim.topology_report
        if report['condition_met'] is False and show_warning:
            messagebox.showwarning(
                'W-MSR graph condition not met',
                f"The sufficient condition {report['required_robustness']} is NOT met.\n\n"
                "The simulation can still run, but the textbook F-local W-MSR guarantee does not apply to this graph."
            )

    def save_run(self):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = Path('simulation') / 'logs' / f'run_{stamp}'
        self.sim.export_csv(out / 'trajectories.csv')
        self.sim.save_figures(out / 'figures')
        from distributed_consensus.evaluation.reporting import write_csv, write_json
        write_csv(out / 'summary.csv', self.sim.summary_rows())
        write_json(out / 'configuration.json', {
            'benchmark': str(self.benchmark),
            'topology': self.sim.topology,
            'seed': self.seed,
            'adversary_ids_1based': [i + 1 for i in self.sim.adversary_ids],
            'F': self.sim.F,
            'fault_model': 'F-local malicious broadcast',
            'attack': 'same automatically scaled extreme scalar signal broadcast to all neighbours',
            'topology_report': self.sim.topology_report,
            'methods': list(METHODS),
            'selected_controller_for_visual_inspection': self.selected_method,
        })
        messagebox.showinfo('Run saved', f'All five controllers were saved to:\n{out.resolve()}')

    def _tick(self):
        now = time.perf_counter()
        elapsed = min(0.25, now - self.last_time)
        self.last_time = now
        if self.running:
            self.accumulator += elapsed * self.rate
            while self.accumulator >= 1.0 and self.sim.k < 250:
                self.sim.step()
                self.accumulator -= 1.0
            if self.sim.k >= 250:
                self.pause()
            else:
                self._refresh()
        self.root.after(40, self._tick)

    def _panel(self, x, y, w, h, title, subtitle=''):
        self.canvas.create_rectangle(x, y, x + w, y + h, fill=PANEL, outline=STROKE, width=1)
        self.canvas.create_text(x + 14, y + 10, text=title, fill=WHITE,
                                font=self._font(18, 'bold'), anchor='nw')
        if subtitle:
            self.canvas.create_text(x + 14, y + 38, text=subtitle, fill=MUTED,
                                    font=self._font(14), anchor='nw',
                                    width=max(120, w - 28), justify='left')

    @staticmethod
    def _coordinates(graph_name, n, x, y, w, h):
        cx, cy = x + w / 2, y + h / 2
        if graph_name == 'chain':
            padding = min(38, w / max(4, 2 * n))
            xs = np.linspace(x + padding, x + w - padding, n)
            return {i: (float(xs[i]), cy) for i in range(n)}
        radius = max(16.0, min(w * .33, h * .37))
        if graph_name == 'star':
            coords = {0: (cx, cy)}
            angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, n - 1, endpoint=False)
            for j, agent in enumerate(i for i in range(n) if i != 0):
                coords[agent] = (cx + radius * np.cos(angles[j]), cy + radius * np.sin(angles[j]))
            return coords
        angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, n, endpoint=False)
        return {i: (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
                for i, angle in enumerate(angles)}

    def _draw_network(self, x, y, w, h):
        report = self.sim.topology_report
        status = 'MET' if report['condition_met'] is True else (
            'NOT MET' if report['condition_met'] is False else 'NOT CHECKED')
        subtitle = f"F={self.sim.F} · {report['required_robustness']} {status}"
        self._panel(x, y, w, h, 'NETWORK', subtitle)
        coords = self._coordinates(self.sim.topology, self.sim.n, x + 15, y + 72,
                                   w - 30, max(55, h - 110))
        adj = self.sim.graph.adjacency.tocsr()
        for sender in range(self.sim.n):
            for pos in range(adj.indptr[sender], adj.indptr[sender + 1]):
                receiver = int(adj.indices[pos])
                if receiver <= sender:
                    continue
                x1, y1 = coords[sender]
                x2, y2 = coords[receiver]
                self.canvas.create_line(x1, y1, x2, y2, fill='#425570', width=1.0)

        # Show only the selected receiver's scalar values discarded by the
        # selected controller. The link remains visible underneath.
        if self.sim.k and self.selected_method.endswith('_wmsr'):
            action = self.sim.action_histories[self.selected_method][-1]['actions'][self.selected_agent]
            if self.selected_agent not in self.sim.adversary_ids:
                for sender in action.removed_ids:
                    sx, sy = coords[sender]
                    rx, ry = coords[self.selected_agent]
                    dx, dy = rx - sx, ry - sy
                    length = max(1.0, math.hypot(dx, dy))
                    ox, oy = -dy / length * 3.0, dx / length * 3.0
                    self.canvas.create_line(
                        sx + ox, sy + oy, rx + ox, ry + oy,
                        fill=ORANGE, width=2.8, dash=(6, 3), arrow=tk.LAST,
                        arrowshape=(10, 11, 4),
                    )

        self.node_hitboxes = []
        for i, (nx, ny) in coords.items():
            compromised = i in self.sim.adversary_ids
            fill = '#9b343f' if compromised else '#28577c'
            outline = RED if compromised else (TEAL if i == self.selected_agent else '#9cb0c7')
            radius = 15 if i == self.selected_agent else 14
            self.canvas.create_oval(nx-radius, ny-radius, nx+radius, ny+radius,
                                    fill=fill, outline=outline, width=2 if i == self.selected_agent else 1.3)
            self.canvas.create_text(nx, ny, text=str(i + 1), fill=WHITE,
                                    font=self._font(14, 'bold'))
            self.node_hitboxes.append((nx-radius-5, ny-radius-5, nx+radius+5, ny+radius+5, i))

        legend_y = y + h - 20
        self.canvas.create_oval(x + 15, legend_y - 7, x + 29, legend_y + 7,
                                fill='#9b343f', outline=RED)
        self.canvas.create_text(x + 36, legend_y, text='Attacker',
                                fill=MUTED, font=self._font(13), anchor='w')
        self.canvas.create_line(x + w * .52, legend_y, x + w * .52 + 28, legend_y,
                                fill=ORANGE, width=2.8, dash=(6, 3), arrow=tk.LAST)
        self.canvas.create_text(x + w * .52 + 37, legend_y, text='Ignored',
                                fill=MUTED, font=self._font(13), anchor='w')

    def _draw_scorecard(self, x, y, w, h):
        self._panel(
            x, y, w, h, 'CONTROLLER RESULTS',
            'e_RMS: current   ·   J_E / J_U: cumulative',
        )
        # Numeric columns are right-aligned to keep their boundaries clear.
        col = {
            'controller': x + 14,
            'error': x + w * .60,
            'J_E': x + w * .80,
            'J_U': x + w - 14,
        }
        zoom_offset = 10.0 * max(0.0, (self.zoom - 1.0) / 0.15)
        header_y = y + 66 + zoom_offset
        for label, key in (
            ('CONTROLLER', 'controller'), ('e_RMS', 'error'),
            ('J_E', 'J_E'), ('J_U', 'J_U'),
        ):
            self.canvas.create_text(
                col[key], header_y, text=label, fill=MUTED,
                font=self._font(13, 'bold'),
                anchor='w' if key == 'controller' else 'e',
            )

        table_top = y + 80 + zoom_offset
        row_h = (h - (table_top - y) - 2) / len(METHODS)
        self.method_hitboxes = []
        for index, method in enumerate(METHODS):
            row_top = table_top + index * row_h
            row_bottom = row_top + row_h
            row_mid = (row_top + row_bottom) / 2
            selected = method == self.selected_method
            if selected:
                self.canvas.create_rectangle(
                    x + 7, row_top + 1, x + w - 7, row_bottom - 1,
                    fill='#203650', outline=METHOD_COLORS[method], width=1.5,
                )
            row = self.sim.latest(method)
            self.canvas.create_text(
                col['controller'], row_mid, text=METHOD_LABELS[method],
                fill=METHOD_COLORS[method], font=self._font(14, 'bold'), anchor='w',
            )
            self.canvas.create_text(
                col['error'], row_mid, text=f"{row['error_rms']:.2e}",
                fill=WHITE, font=self._font(14), anchor='e',
            )
            self.canvas.create_text(
                col['J_E'], row_mid, text=f"{row['disagreement_cost']:.3g}",
                fill=WHITE, font=self._font(14), anchor='e',
            )
            self.canvas.create_text(
                col['J_U'], row_mid, text=f"{row['energy']:.3g}",
                fill=WHITE, font=self._font(14), anchor='e',
            )
            self.method_hitboxes.append((x + 5, row_top, x + w - 5, row_bottom, method))

    def _draw_error_chart(self, x, y, w, h):
        self._panel(
            x, y, w, h, 'NORMAL-AGENT DISAGREEMENT',
            'e_RMS over time · all five controllers · log scale',
        )
        left, right = x + 88, x + w - 18
        top, bottom = y + 60, y + h - 43
        if right <= left or bottom <= top:
            return
        values = np.asarray([
            max(float(value), 1e-12)
            for method in METHODS for value in self.sim.error_histories[method]
        ])
        low = min(-8.0, float(np.floor(np.log10(values.min()))) - .3)
        high = max(0.5, float(np.ceil(np.log10(values.max())) + .2)) 
        tick_values = np.unique(np.round(np.linspace(high, low, 5)).astype(int))
        for exponent in tick_values:
            if exponent < low or exponent > high:
                continue
            yy = top + (high - exponent) / (high - low) * (bottom - top)
            self.canvas.create_line(left, yy, right, yy, fill='#2a3b51', dash=(2, 4))
            self.canvas.create_text(
                left - 9, yy, text=f'10^{int(exponent)}', fill=MUTED,
                font=self._font(12), anchor='e',
            )
        self.canvas.create_line(left, top, left, bottom, fill='#637a94')
        self.canvas.create_line(left, bottom, right, bottom, fill='#637a94')
        self.canvas.create_text(
            left, bottom + 12, text='0', fill=MUTED,
            font=self._font(12), anchor='w',
        )
        self.canvas.create_text(
            right, bottom + 12, text=str(self.sim.k), fill=MUTED,
            font=self._font(12), anchor='e',
        )
        if low <= math.log10(self.sim.threshold) <= high:
            yy = top + (high - math.log10(self.sim.threshold)) / (high - low) * (bottom - top)
            self.canvas.create_line(left, yy, right, yy, fill=WHITE, dash=(6, 4), width=1.4)
            self.canvas.create_text(
                right - 4, yy - 10, text=f'threshold {self.sim.threshold:g}',
                fill=WHITE, font=self._font(12), anchor='e',
            )
        for method in METHODS:
            stream = np.maximum(np.asarray(self.sim.error_histories[method], dtype=float), 1e-12)
            points = []
            end = max(1, self.sim.k)
            for step, value in enumerate(stream):
                xx = left + step / end * (right - left)
                yy = top + (high - math.log10(value)) / (high - low) * (bottom - top)
                points.extend((xx, yy))
            if len(points) >= 4:
                self.canvas.create_line(*points, fill=METHOD_COLORS[method], width=2.4)
        legend_y = y + h - 15
        legend = [
            ('Local', 'local_only'), ('Paper raw', 'paper_gnn_raw'),
            ('Paper + W-MSR', 'paper_gnn_wmsr'), ('SG raw', 'small_gain_raw'),
            ('SG + W-MSR', 'small_gain_wmsr'),
        ]
        cell = (right - left) / len(legend)
        for index, (label, method) in enumerate(legend):
            lx = left + index * cell
            self.canvas.create_line(
                lx, legend_y, lx + 20, legend_y,
                fill=METHOD_COLORS[method], width=2.8,
            )
            self.canvas.create_text(
                lx + 26, legend_y, text=label, fill=MUTED,
                font=self._font(12), anchor='w',
            )

    @staticmethod
    def _ids(ids):
        return ', '.join(str(value + 1) for value in ids) if ids else 'none'

    def _show_agent_messages(self, agent: int):
        """Open a readable audit for the selected receiver and method."""
        self.selected_agent = int(agent)
        self._refresh()
        win = tk.Toplevel(self.root)
        win.title(f'Agent {agent + 1} — received messages')
        win.geometry('820x680')
        win.minsize(720, 560)
        win.configure(bg=BG)
        outer = tk.Frame(win, bg=BG, padx=16, pady=14)
        outer.pack(fill='both', expand=True)
        state = self.sim.states[self.selected_method][agent]
        compromised = agent in self.sim.adversary_ids
        role = 'MALICIOUS AGENT' if compromised else 'NORMAL AGENT'
        tk.Label(
            outer, text=f'Agent {agent + 1} · {role}', bg=BG, fg=WHITE,
            font=self._font(20, 'bold'), anchor='w'
        ).pack(fill='x', pady=(0, 4))
        tk.Label(
            outer, text=f"Controller: {METHOD_LABELS[self.selected_method]}",
            bg=BG, fg=MUTED, font=self._font(14), anchor='w', justify='left', wraplength=760
        ).pack(fill='x')
        tk.Label(
            outer, text=f"Physical state: [{state[0]:+.4f}, {state[1]:+.4f}]    Step: {self.sim.k}",
            bg=BG, fg=MUTED, font=self._font(14), anchor='w', justify='left', wraplength=760
        ).pack(fill='x', pady=(2, 12))
        text = tk.Text(
            outer, wrap='word', bg=PANEL, fg=WHITE, relief='flat',
            font=self._font(15), padx=16, pady=14, spacing1=4, spacing3=7
        )
        scroll = ttk.Scrollbar(outer, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        if self.sim.k == 0:
            audit = 'No messages have been processed yet. Press STEP or PLAY first.'
        else:
            action = self.sim.action_histories[self.selected_method][-1]['actions'][agent]
            messages = '\n'.join(
                f'Agent {sender + 1}: {value:+.5f}'
                for sender, value in action.neighbor_messages
            ) or 'No neighbours.'
            if compromised:
                attack = self.sim.action_histories[self.selected_method][-1]['attack_value']
                decision = (
                    f'This agent is the compromised sender. It broadcast {attack:+.5f} '
                    'to its neighbours and does not run the W-MSR filter on its own inbox.'
                )
            elif self.selected_method.endswith('_wmsr'):
                message_map = dict(action.neighbor_messages)
                retained_values = [message_map[j] for j in action.retained_ids]
                wmsr_output = float(np.mean([action.scalar_signal, *retained_values]))
                decision = (
                    f"My scalar signal: {action.scalar_signal:+.5f}\n"
                    f"Kept neighbour IDs: {self._ids(action.retained_ids)}\n"
                    f"Removed highest values above mine: {self._ids(action.removed_high_ids)}\n"
                    f"Removed lowest values below mine: {self._ids(action.removed_low_ids)}\n"
                    f"W-MSR mean (my value + kept values): {wmsr_output:+.5f}"
                )
            else:
                decision = f"My scalar signal: {action.scalar_signal:+.5f}\nNo W-MSR filtering: every neighbour message is used."
            audit = f'MY NEIGHBOUR MESSAGES\n{messages}\n\nW-MSR RESULT\n{decision}'
        text.insert('1.0', audit)
        text.configure(state='disabled')

    def _refresh(self):
        if not hasattr(self, 'canvas'):
            return
        self.canvas.delete('all')
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 300 or height < 250:
            return
        self.step_label.configure(text=f'Step {self.sim.k} / 250')

        margin, gap = 8, 10
        usable_w = width - 2 * margin
        left_w = int((usable_w - gap) * .34)
        right_x = margin + left_w + gap
        right_w = usable_w - left_w - gap
        usable_h = height - 2 * margin - gap
        top_h = min(225, max(205, int(usable_h * .52)))
        chart_h = max(150, usable_h - top_h - gap)
        self._draw_network(margin, margin, left_w, top_h)
        self._draw_scorecard(right_x, margin, right_w, top_h)
        chart_y = margin + top_h + gap
        self._draw_error_chart(margin, chart_y, usable_w, chart_h)

    def _mouse_down(self, event):
        for x1, y1, x2, y2, method in self.method_hitboxes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.selected_method = method
                self._refresh()
                return
        for x1, y1, x2, y2, agent in self.node_hitboxes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._show_agent_messages(agent)
                return


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Scalar local W-MSR multi-controller simulation')
    parser.add_argument('--benchmark', default=str(DEFAULT_BENCHMARK_CHECKPOINT))
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args(argv)
    root = tk.Tk()
    DemoApp(root, benchmark=args.benchmark, seed=args.seed)
    root.mainloop()


if __name__ == '__main__':
    main()
