"""Tkinter-only interactive presentation GUI for the independent MAS simulator.

All physics are delegated to model.py; this file only draws actual state arrays
and receives operator commands. No rewrites of the research package are needed.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import math
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from .model import DualSimulation, DEFAULT_CHECKPOINT, TOPOLOGIES, DT, DISAGREEMENT_TOL

BG = '#0b1220'
PANEL = '#152235'
PANEL_2 = '#1c2b41'
STROKE = '#293950'
WHITE = '#edf4fc'
MUTED = '#9caec2'
ORANGE = '#ffae78'
TEAL = '#58dfc2'
BLUE = '#77a6ed'
RED = '#ef7e87'
FNT = 'DejaVu Sans'


def safe_float(text, default):
    try:
        x = float(text)
        if np.isfinite(x):
            return x
    except (ValueError, TypeError):
        pass
    return default


class DemoApp:
    def __init__(self, root, checkpoint=DEFAULT_CHECKPOINT, seed=3):
        self.root = root
        self.checkpoint = Path(checkpoint)
        self.sim = DualSimulation(self.checkpoint, seed=seed)
        self.selected = 2  # Zero-based, third actuator
        self.running = False
        self.rate = 12
        self._last_clock = time.perf_counter()
        self._accumulator = 0.0
        self._drag_start = None
        self._drag_delta = 0.0
        self._scene_hitboxes = []
        self._layout()
        self._refresh()
        root.after(35, self._tick)

    def _button(self, parent, label, callback, accent=False):
        bg = '#237f75' if accent else PANEL_2
        return tk.Button(parent, text=label, command=callback, font=(FNT, 10, 'bold'),
                         bg=bg, fg=WHITE, activebackground='#36536a',
                         activeforeground=WHITE, relief='flat', bd=0, padx=12,
                         pady=8, highlightthickness=0, cursor='hand2')

    def _layout(self):
        root = self.root
        root.title('MAS Control Lab | Multi-Actuator Alignment')
        root.geometry('1510x940')
        root.minsize(1120, 760)
        root.configure(bg=BG)
        top = tk.Frame(root, bg=BG, pady=13, padx=20)
        top.pack(fill='x')
        tk.Label(top, text='CONTROL LAB', bg=BG, fg=TEAL,
                 font=(FNT, 10, 'bold')).pack(anchor='w')
        tk.Label(top, text='Coordinated multi-actuator alignment', bg=BG,
                 fg=WHITE, font=(FNT, 20, 'bold')).pack(anchor='w')
        tk.Label(top, text='Physical 1-DOF actuators  •  Same system / same events  •  Local-only vs learned neighbor feedback',
                 bg=BG, fg=MUTED, font=(FNT, 10)).pack(anchor='w')

        bar = tk.Frame(root, bg=BG, padx=20)
        bar.pack(fill='x', pady=(0, 5))
        self.play_button = self._button(bar, '▶  PLAY', self.play_pause, True)
        self.play_button.pack(side='left', padx=(0, 6))
        self._button(bar, 'STEP', self.advance).pack(side='left', padx=5)
        self._button(bar, '↺  RESTART', self.restart).pack(side='left', padx=5)
        self._button(bar, 'SAVE CSV', self.save_csv).pack(side='left', padx=5)

        self.step_label = tk.Label(bar, text='k = 000   |   t = 0.0', bg=BG, fg=WHITE,
                                   font=(FNT, 11, 'bold'))
        self.step_label.pack(side='left', padx=18)
        tk.Label(bar, text='Steps/sec', bg=BG, fg=MUTED).pack(side='left')
        self.speed_var = tk.IntVar(value=self.rate)
        speed = tk.Scale(bar, from_=2, to=45, orient='horizontal', variable=self.speed_var,
                         bg=BG, fg=WHITE, troughcolor=STROKE, highlightthickness=0,
                         showvalue=True, length=145, bd=0)
        speed.pack(side='left')
        self.advantage_label = tk.Label(bar, text='BOTH ACTIVE', bg=BG, fg=TEAL,
                                        font=(FNT, 11, 'bold'))
        self.advantage_label.pack(side='right', padx=12)

        options = tk.Frame(root, bg=BG, padx=20)
        options.pack(fill='x', pady=(3, 8))
        tk.Label(options, text='Agents', bg=BG, fg=MUTED).pack(side='left')
        self.n_var = tk.StringVar(value='10')
        self._combo(options, self.n_var, ('3', '5', '10', '20'), 4).pack(side='left', padx=(8, 16))
        tk.Label(options, text='Graph', bg=BG, fg=MUTED).pack(side='left')
        self.graph_var = tk.StringVar(value='checkpoint')
        self._combo(options, self.graph_var, ('checkpoint',) + TOPOLOGIES, 13).pack(
            side='left', padx=(8, 16))
        tk.Label(options, text='Seed', bg=BG, fg=MUTED).pack(side='left')
        self.seed_var = tk.StringVar(value=str(self.sim.seed))
        tk.Entry(options, textvariable=self.seed_var, width=5, bg=PANEL_2,
                 fg=WHITE, insertbackground=WHITE, relief='flat', font=(FNT, 11)).pack(
                     side='left', padx=(8, 10))
        self._button(options, 'APPLY / RESET', self.apply_config).pack(side='left')
        tk.Label(options, text='    Selected actuator', bg=BG, fg=MUTED).pack(side='left')
        self.selected_label = tk.Label(options, text='#03', bg=BG, fg=TEAL,
                                       font=(FNT, 11, 'bold'))
        self.selected_label.pack(side='left', padx=5)
        tk.Label(options, text='Pulse amplitude', bg=BG, fg=MUTED).pack(side='left', padx=(14, 5))
        self.strength_var = tk.StringVar(value='8.0')
        tk.Entry(options, textvariable=self.strength_var, width=5, bg=PANEL_2,
                 fg=WHITE, insertbackground=WHITE, relief='flat', font=(FNT, 11)).pack(side='left')
        self._button(options, '⚡ PULSE BOTH', self.inject_pulse, True).pack(side='left', padx=10)

        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0, bd=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True, padx=16, pady=(0, 8))
        self.canvas.bind('<ButtonPress-1>', self._mouse_down)
        self.canvas.bind('<B1-Motion>', self._mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self._mouse_up)
        self.canvas.bind('<Configure>', lambda _e: self._refresh())
        footer = tk.Frame(root, bg=BG, padx=20, pady=8)
        footer.pack(fill='x')
        self.status = tk.Label(footer, text='Click a column to select it. Drag vertically to displace that actuator in BOTH simulations.',
                               fg=MUTED, bg=BG, font=(FNT, 10), anchor='w')
        self.status.pack(side='left', fill='x', expand=True)
        tk.Label(footer, text='No rigid-platform mechanics  •  No actuator saturation',
                 bg=BG, fg='#6d879e', font=(FNT, 9)).pack(side='right')
        root.bind('<space>', lambda e: self.play_pause())
        root.bind('<Right>', lambda e: self.advance())
        root.bind('<Escape>', lambda e: self.pause())
        root.bind('<Control-r>', lambda e: self.restart())

    def _combo(self, parent, variable, values, width):
        combo = ttk.Combobox(parent, textvariable=variable, values=values, width=width,
                             state='readonly')
        return combo

    def play_pause(self):
        self.running = not self.running
        self.play_button.configure(text='Ⅱ  PAUSE' if self.running else '▶  PLAY')
        self._last_clock = time.perf_counter()
        self._accumulator = 0.

    def pause(self):
        if self.running:
            self.play_pause()

    def advance(self):
        self.pause()
        self.sim.step()
        self._refresh()

    def restart(self):
        self.pause()
        self.sim.reset()
        self.status.configure(text='Restarted; identical saved initial states for both controllers.')
        self._refresh()

    def apply_config(self):
        try:
            n = int(self.n_var.get())
            seed = int(self.seed_var.get())
            topology = self.graph_var.get()
            candidate = DualSimulation(self.checkpoint, n=n, topology=topology, seed=seed)
        except (ValueError, OSError, KeyError) as exc:
            messagebox.showerror('Invalid configuration', str(exc))
            return
        self.pause()
        self.sim = candidate
        self.selected = min(self.selected, n - 1)
        self.status.configure(text=('Original 10-agent physical system loaded.' if n == 10 else
                'Changed N: existing plant generator used; frozen W transferred, not retrained. '
                'Performance may differ.'))
        self._refresh()

    def inject_pulse(self):
        magnitude = safe_float(self.strength_var.get(), float('nan'))
        if not np.isfinite(magnitude) or abs(magnitude) > 10000:
            messagebox.showerror('Invalid pulse', 'Enter a finite pulse amplitude with |value| <= 10000.')
            return
        self.sim.pulse(self.selected, magnitude, duration=3)
        self.status.configure(text=f'Identical actuator-input pulse on agent #{self.selected + 1}: '
                                   f'{magnitude:+g} for 3 steps. Readiness counters reset.')
        self._refresh()

    def save_csv(self):
        if self.sim.k < 1:
            self.status.configure(text='Play or STEP at least once before exporting a history.')
            return
        out = Path.cwd() / 'simulation' / 'logs' / (
            f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            self.sim.export_csv(out)
        except OSError as exc:
            messagebox.showerror('Export error', str(exc))
            return
        self.status.configure(text=f'CSV saved: {out}')

    def _tick(self):
        clock = time.perf_counter()
        dt = min(clock - self._last_clock, 0.3)
        self._last_clock = clock
        if self.running:
            self._accumulator += dt * self.speed_var.get()
            count = min(int(self._accumulator), 15)
            for _ in range(count):
                self.sim.step()
            self._accumulator -= count
            if count:
                self._refresh()
        self.root.after(34, self._tick)

    def _mouse_down(self, event):
        for x1, y1, x2, y2, lane_positions, px_per_q in self._scene_hitboxes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2 and lane_positions:
                self.selected = int(np.argmin([abs(event.x - x) for x in lane_positions]))
                self._drag_start = (event.y, px_per_q)
                self._drag_delta = 0.
                self._refresh()
                return
        self._drag_start = None

    def _mouse_move(self, event):
        if self._drag_start is not None:
            y0, scale = self._drag_start
            self._drag_delta = float(np.clip((y0 - event.y) / scale, -8., 8.))
            self.status.configure(text=f'Drag preview: agent #{self.selected + 1}, '
                                       f'position displacement {self._drag_delta:+.2f}. '
                                       'Release to apply the SAME delta to both worlds.')

    def _mouse_up(self, event):
        if self._drag_start is not None:
            y0, scale = self._drag_start
            delta = float(np.clip((y0 - event.y) / scale, -8., 8.))
            if abs(event.y - y0) >= 5:
                self.sim.drag(self.selected, delta)
                self.status.configure(text=f'Manual position displacement {delta:+.2f} on '
                                           f'actuator #{self.selected + 1} in BOTH worlds.')
            self._drag_start = None
            self._drag_delta = 0.
            self._refresh()

    def _txt(self, x, y, txt, color=WHITE, size=10, weight='normal', anchor='nw'):
        self.canvas.create_text(x, y, text=txt, fill=color, font=(FNT, size, weight), anchor=anchor)

    def _box(self, x1, y1, x2, y2, fill=PANEL, outline=STROKE):
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)

    def _line(self, coords, color, width=2, dash=None):
        self.canvas.create_line(*coords, fill=color, width=width, dash=dash, smooth=False)

    def _refresh(self):
        if not hasattr(self, 'canvas'):
            return
        c = self.canvas
        w, h = max(c.winfo_width(), 1090), max(c.winfo_height(), 475)
        c.delete('all')
        self.step_label.configure(text=f'k = {self.sim.k:03d}   |   model time = {self.sim.k * DT:.1f}')
        self.selected_label.configure(text=f'#{self.selected + 1:02d}')
        ta, tb = self.sim.ready_step['baseline'], self.sim.ready_step['learned']
        if ta is not None and tb is not None:
            if tb < ta:
                advantage = f'LEARNED READY {ta - tb} STEPS SOONER'
            elif ta < tb:
                advantage = f'BASELINE READY {tb - ta} STEPS SOONER'
            else:
                advantage = 'SAME READINESS TIME'
        elif tb is not None:
            advantage = 'LEARNED READY FIRST'
        elif ta is not None:
            advantage = 'BASELINE READY FIRST'
        else:
            advantage = 'BOTH ACTIVE'
        self.advantage_label.configure(text=advantage)
        gap = 16
        margin = 6
        scene_w = (w - 2 * margin - gap) / 2
        scene_top = 8
        scene_bottom = max(285., min(h * .56, h - 238))
        self._scene_hitboxes = []
        both_states = np.concatenate([self.sim.worlds[mode].states[:, 0]
                                      for mode in ('baseline', 'learned')])
        scale = min(39., 83. / max(.2, float(np.max(np.abs(both_states)))))
        for index, (mode, title, tint) in enumerate((('baseline', 'BASELINE  /  LOCAL ONLY', ORANGE),
                                                    ('learned', 'LEARNED  /  NEIGHBOR FEEDBACK', TEAL))):
            left = margin + index * (scene_w + gap)
            self._draw_scene(left, scene_top, scene_w, scene_bottom - scene_top,
                             mode, title, tint, scale)
        lower_top = scene_bottom + 14
        lower_bottom = h - 9
        lower_h = lower_bottom - lower_top
        graph_w = max(285, min(scene_w * .7, w * .38))
        self._draw_graph(margin, lower_top, graph_w, lower_h)
        chart_x = margin + graph_w + gap
        chart_w = w - margin - chart_x
        self._draw_chart(chart_x, lower_top, chart_w, lower_h)

    def _draw_scene(self, x, y, w, h, mode, title, tint, scale):
        self._box(x, y, x + w, y + h)
        self._box(x + 1, y + 1, x + 5, y + h - 1, fill=tint, outline=tint)
        self._txt(x + 20, y + 12, title, tint, 11, 'bold')
        r = self.sim.latest(mode)
        label = f'READINESS  k={r["ready_step"]}' if r['ready_step'] is not None else 'READINESS  waiting'
        if r['ready_step'] is not None and self.sim.segment_start:
            label += f'  |  recovery {r["ready_step"]-self.sim.segment_start} steps'
        self._txt(x + 20, y + 36, label, WHITE, 11, 'bold')
        self._txt(x + 20, y + 60,
                  f'e RMS  {r["error_rms"]:.4g}     |     height spread  {r["height_spread"]:.3f}'
                  f'     |     max |q|  {r["position_max"]:.3f}', MUTED, 9)
        ground = y + h - 62
        zero = ground - min(118, max(91, h * .35))
        self._line([x + 14, zero, x + w - 14, zero], '#5c8097', 1, (5, 6))
        self._txt(x + w - 18, zero - 16, 'target q = 0', MUTED, 8, anchor='ne')
        self._line([x + 16, ground, x + w - 16, ground], '#6f8da5', 3)
        n = self.sim.n
        lane_gap = (w - 72) / (n - 1) if n > 1 else 0
        xs = [x + 36 + i * lane_gap for i in range(n)]
        q = self.sim.worlds[mode].states[:, 0]
        ys = [zero - float(q[i]) * scale for i in range(n)]
        barw = min(24., max(8., lane_gap * .48))
        for i, (cx, cy) in enumerate(zip(xs, ys)):
            selected = i == self.selected
            self._line([cx, cy, cx, ground], STROKE, max(4., barw - 4.))
            self._line([cx, cy + 4, cx, ground - 4], '#476078', max(2., barw - 10.))
            if cy < ground:
                c = self.canvas
                c.create_rectangle(cx - barw / 2, cy, cx + barw / 2, ground,
                                   fill='#324a5d' if not selected else '#3b6880', outline=STROKE)
            self.canvas.create_rectangle(cx - barw * .67, cy - 4, cx + barw * .67, cy + 5,
                                         fill=tint if not selected else '#ffe1a6', outline='')
            if selected:
                self.canvas.create_oval(cx - 9, cy - 11, cx + 9, cy + 9,
                                        outline=WHITE, width=1)
            self._txt(cx, ground + 9, str(i + 1), WHITE if selected else MUTED,
                      8, 'bold' if selected else 'normal', anchor='n')
        # Virtual alignment guide connects *rendered observations*, not rigid mechanics.
        for i in range(n - 1):
            self._line((xs[i], ys[i], xs[i + 1], ys[i + 1]), tint, 1, (3, 4))
        self._txt(x + 18, ground - 4, '0', MUTED, 8)
        self._scene_hitboxes.append((x + 10, y + 84, x + w - 10, ground + 20, xs, scale))
        self._txt(x + 18, y + h - 16, f'J_E = {r["disagreement_cost"]:.2f}     '
                  f'J_U = {r["energy"]:.2f}', tint, 9, 'bold', anchor='sw')
        self._txt(x + w - 16, y + h - 13, 'dashed line = virtual alignment only', MUTED, 8, anchor='se')

    def _draw_graph(self, x, y, w, h):
        self._box(x, y, x + w, y + h)
        self._txt(x + 18, y + 12, 'COMMUNICATION NETWORK', WHITE, 11, 'bold')
        self._txt(x + 18, y + 36, f'{self.sim.n} nodes  •  {"chain (fallback)" if self.sim.topology == "checkpoint" and self.sim.n != 10 else self.sim.topology}  •  undirected', MUTED, 9)
        n = self.sim.n
        cx, cy = x + w * .48, y + h * .57
        rad = max(38., min(w * .27, h * .28))
        coords = [(cx + rad * math.cos(2 * math.pi * i / n - math.pi / 2),
                   cy + rad * math.sin(2 * math.pi * i / n - math.pi / 2)) for i in range(n)]
        adj = self.sim.graph.adjacency.tocsr()
        for i in range(n):
            for j in adj.indices[adj.indptr[i]:adj.indptr[i + 1]]:
                if j > i:
                    self._line((*coords[i], *coords[j]), '#45617e', 1.6)
        neighbors = set(adj.indices[adj.indptr[self.selected]:adj.indptr[self.selected + 1]])
        for i, (px, py) in enumerate(coords):
            selected = i == self.selected
            fill = '#f5c28d' if selected else ('#568cad' if i in neighbors else '#254560')
            self.canvas.create_oval(px - 13, py - 13, px + 13, py + 13, fill=fill,
                                    outline=TEAL if selected else '#5e7992', width=2 if selected else 1)
            self._txt(px, py, str(i + 1), BG if selected else WHITE, 9, 'bold', 'center')
        self._txt(x + 15, y + h - 52, f'SELECTED AGENT #{self.selected + 1:02d}', TEAL, 10, 'bold')
        a = self.sim.latest('baseline')
        b = self.sim.latest('learned')
        def signed(num):
            return '—' if num is None else f'{num:+.3f}'
        for i, (name, item, col) in enumerate((('BASE', a, ORANGE), ('LEARNED', b, TEAL))):
            states = item['states'][self.selected]
            last = item['last_action']
            u = None if last is None else float(last['total'][self.selected, 0])
            v = None if last is None else float(last['correction'][self.selected, 0])
            self._txt(x + 15, y + h - 31 + 15 * i,
                      f'{name:7} q {states[0]:+.2f}   vel {states[1]:+.2f}   '
                      f'v {signed(v)}   u {signed(u)}', col, 8)

    def _draw_chart(self, x, y, w, h):
        self._box(x, y, x + w, y + h)
        self._txt(x + 18, y + 12, 'LIVE DISAGREEMENT', WHITE, 11, 'bold')
        self._txt(x + w - 16, y + 13, 'random-walk Laplacian | RMS (log scale)', MUTED, 8,
                  anchor='ne')
        self._txt(x + 18, y + 35, 'LOCAL ONLY', ORANGE, 9, 'bold')
        self._txt(x + 114, y + 35, 'LEARNED', TEAL, 9, 'bold')
        left, right = x + 64, x + w - 20
        top, bottom = y + 65, y + h - 40
        if bottom <= top:
            return
        self._line([left, top, left, bottom, right, bottom], '#50657b', 1)
        streams = [self.sim.error_histories[name] for name in ('baseline', 'learned')]
        end_k = self.sim.k
        begin_k = max(0, end_k - 180)
        displayed = np.array([max(value, 1e-9) for s in streams for value in s[begin_k:]],
                             dtype=float)
        if len(displayed) == 0:
            return
        high = max(0.2, float(np.max(np.log10(displayed))) + 0.2)
        low = min(-4.5, float(np.min(np.log10(displayed))) - .3)
        span = max(20, end_k - begin_k)
        def xx(k): return left + (k - begin_k) / span * (right - left)
        def yy(v): return top + (high - math.log10(max(v, 1e-9))) / (high - low) * (bottom - top)
        for logvalue in (0., -1., -2., -3., -4.):
            if low <= logvalue <= high:
                yp = top + (high - logvalue) / (high - low) * (bottom - top)
                self._line([left, yp, right, yp], '#30445a', 1, (3, 6))
                self._txt(left - 8, yp, f'10^{int(logvalue)}', MUTED, 8, anchor='e')
        for event_k, label in self.sim.events:
            if begin_k <= event_k <= end_k:
                pos = xx(event_k)
                self._line([pos, top, pos, bottom], '#916fb0', 1, (5, 5))
                self._txt(pos + 4, top + 4, 'EVENT', '#bc93dd', 8)
        for mode, tint in (('baseline', ORANGE), ('learned', TEAL)):
            values = self.sim.error_histories[mode]
            if end_k > begin_k:
                pts = []
                for k in range(begin_k, end_k + 1):
                    pts += [xx(k), yy(values[k])]
                if len(pts) >= 4:
                    self._line(pts, tint, 2.4)
        self._txt(left, bottom + 9, f'k={begin_k}', MUTED, 8)
        self._txt(right, bottom + 9, f'k={end_k}', MUTED, 8, anchor='ne')
        a, b = self.sim.latest('baseline'), self.sim.latest('learned')
        def tag(r):
            return '—' if r['error_step'] is None else str(r['error_step'])
        self._txt(x + 16, y + h - 15,
                  f'Sustained RMS ≤ {DISAGREEMENT_TOL:g}  |  '
                  f'BASE k={tag(a)}     LEARNED k={tag(b)}', WHITE, 9, 'bold', anchor='sw')


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Interactive distributed MAS visual demonstration')
    p.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT),
                   help='Original project checkpoint (.npz); numerical-only')
    p.add_argument('--seed', type=int, default=3)
    args = p.parse_args(argv)
    root = tk.Tk()
    DemoApp(root, checkpoint=args.checkpoint, seed=args.seed)
    root.mainloop()


if __name__ == '__main__':
    main()
