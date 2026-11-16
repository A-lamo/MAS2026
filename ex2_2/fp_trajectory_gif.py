"""Exercise 2.2 - animated fictitious play on the best-response triangles (GIF).

Same runs as fp_trajectory_plot.py. Frames are spaced logarithmically in the round
number, since fictitious-play steps shrink like 1/t and the interesting moves are early.
"""
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from best_response_plot import A, B, COLS, ROWS, draw_panel
from fp_trajectory_plot import ITER, LINE_COLORS, eqs, starts, trajectory

FRAMES, HOLD, FPS = 110, 20, 12
rounds = np.unique(np.geomspace(1, ITER, FRAMES).astype(int)) - 1   # round index per frame
rounds = np.concatenate([rounds, np.full(HOLD, rounds[-1])])          # pause on last frame

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.6))
names = [f"NE{k + 1}" for k in range(len(eqs))]
draw_panel(ax1, A, ROWS, COLS, "Column's empirical mix y  (regions: row's best response)",
           [(n, y) for n, (x, y) in zip(names, eqs)])
draw_panel(ax2, B.T, COLS, ROWS, "Row's empirical mix x  (regions: column's best response)",
           [(n, x) for n, (x, y) in zip(names, eqs)])

halo = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
artists = []   # (line, head, points) for every run in both panels
for (name, init), color in zip(starts.items(), LINE_COLORS):
    xs, ys = trajectory(init)
    for ax, pts in ((ax1, ys), (ax2, xs)):
        ax.plot(*pts[0], "o", color=color, ms=6, mec="white", zorder=7)
        line, = ax.plot([], [], color=color, lw=1.6, zorder=6, label=name, path_effects=halo)
        head, = ax.plot([], [], "o", color=color, ms=10, mec="black", mew=1.2, zorder=8)
        artists.append((line, head, pts))

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10)
counter = fig.text(0.5, 0.115, "", ha="center", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.15, 1, 0.97), w_pad=4)


def update(frame):
    t = rounds[frame]
    for line, head, pts in artists:
        line.set_data(*pts[:t + 1].T)
        head.set_data([pts[t, 0]], [pts[t, 1]])
    counter.set_text(f"round {t + 1} / {ITER}")
    return [a for line, head, _ in artists for a in (line, head)] + [counter]


out = Path(__file__).with_name("fp_trajectories.gif")
FuncAnimation(fig, update, frames=len(rounds), blit=False).save(
    out, writer=PillowWriter(fps=FPS), dpi=80)
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(rounds)} frames)")
