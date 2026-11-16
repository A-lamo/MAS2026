"""Exercise 2.2 - animated version of fp_corner_batches.py (GIF).

Same runs as the static plot. Each run is a moving dot with a fading trail; frames
are spaced logarithmically in the round number, since fictitious-play steps shrink
like 1/t and the interesting moves happen early. animate() is reused by fp_cluster_gif.py.
"""
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from fp_corner_batches import BATCH_COLORS, ITER, N, batch_runs, setup_figure
from fp_trajectory_plot import RIGHT

FRAMES, HOLD, FPS = 110, 20, 12


def animate(fig, ax1, ax2, tracks, out, head_size=28, start_size=2.5, trail_width=0.8,
            trail_alpha=0.35, head_edge=0.6):
    """tracks: [(ys, xs, color_left, color_right), ...] with ys/xs the empirical mixes
    per round of column (left panel) and row (right panel)."""
    n_rounds = len(tracks[0][0])
    rounds = np.unique(np.geomspace(1, n_rounds, FRAMES).astype(int)) - 1
    rounds = np.concatenate([rounds, np.full(HOLD, rounds[-1])])     # pause on last frame

    lines, pts, cols = [], {ax1: [], ax2: []}, {ax1: [], ax2: []}
    for ys, xs, c_left, c_right in tracks:
        for ax, mix, color in ((ax1, ys, c_left), (ax2, xs, c_right)):
            p = mix @ RIGHT
            if start_size:                                          # 0 = no start markers
                ax.plot(*p[0], "o", color=color, ms=start_size, alpha=0.6, zorder=7)
            line, = ax.plot([], [], color=color, lw=trail_width, alpha=trail_alpha, zorder=6)
            lines.append((line, p))
            pts[ax].append(p); cols[ax].append(color)
    heads = {}
    for ax in (ax1, ax2):
        pts[ax] = np.stack(pts[ax])                                   # (runs, rounds, 2)
        heads[ax] = ax.scatter(*pts[ax][:, 0].T, s=head_size, c=cols[ax], edgecolors="white",
                               linewidths=head_edge, zorder=8)
    counter = fig.text(0.5, 0.95, "", ha="center", fontsize=13, fontweight="bold")

    def update(frame):
        t = rounds[frame]
        for line, p in lines:
            line.set_data(*p[:t + 1].T)
        for ax in (ax1, ax2):
            heads[ax].set_offsets(pts[ax][:, t])
        counter.set_text(f"round {t} / {n_rounds - 1}")             # index 0 = the prior

    FuncAnimation(fig, update, frames=len(rounds)).save(
        out, writer=PillowWriter(fps=FPS), dpi=70)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(rounds)} frames)")


if __name__ == "__main__":
    fig, ax1, ax2 = setup_figure(f"{N} players per batch x 9 batch pairs;  o = start")
    tracks = [(ys, xs, BATCH_COLORS[ci], BATCH_COLORS[ri]) for ri, ci, xs, ys in batch_runs()]
    animate(fig, ax1, ax2, tracks, Path(__file__).with_name("fp_corner_batches.gif"))
