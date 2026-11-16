"""Exercise 2.2 - fictitious play trajectories on the best-response triangles.

Each round, a player best-responds to the opponent's empirical mix, i.e. plays the
action whose region (left: row's regions over column's mix, right: column's regions
over row's mix) the opponent's current empirical mix lies in. Its own empirical mix
then takes a 1/t step towards that action's corner in the other triangle.

Saves two versions of the same runs:
  fp_trajectories.png        - equilateral simplices
  fp_trajectories_right.png  - right triangles in the (p, q) and (i, j) planes, with
                               y = (p, q, r = 1-p-q) and x = (i, j, k = 1-i-j)
"""
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np

from best_response_plot import A, B, CORNERS, COLS, ROWS, draw_panel

# (p, q) plane: L = (1, 0), C = (0, 1), R = origin.  Same for (i, j): T, M, B.
RIGHT = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

np.random.seed(0)             # nashpy breaks best-response ties with np.random
ITER = 3000
game = nash.Game(A, B)
eqs = list(game.support_enumeration())


def trajectory(init, iterations=ITER, corners=CORNERS):
    """Empirical mixes (x_t, y_t) after every round, starting from prior counts init,
    mapped to plot coordinates via corners."""
    counts = tuple(np.array(c, float) for c in init)
    xs, ys = [], []
    for r, c in game.fictitious_play(iterations=iterations, play_counts=counts):
        xs.append(r / r.sum()); ys.append(c / c.sum())
    return np.array(xs) @ corners, np.array(ys) @ corners


rng = np.random.default_rng(1)
starts = {
    "start T,C": ((1, 0, 0), (0, 1, 0)),
    "start M,R": ((0, 1, 0), (0, 0, 1)),
    "uniform": ((1, 1, 1), (1, 1, 1)),
    "random": tuple(rng.integers(1, 10, 3) for _ in range(2)),
    "near NE2 (prior 30)": tuple(np.round(30 * v) for v in eqs[1]),
    "near NE3 (prior 30)": tuple(np.round(30 * v) for v in eqs[2]),
}
LINE_COLORS = ["#FF6F00", "#8E44AD", "#C0392B", "#7F5A00", "#D81B60", "#00838F"]


def make_plot(corners, out, axis_labels=(None, None)):
    np.random.seed(0)         # same tie-breaks, so both versions show identical runs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.8))
    names = [f"NE{k + 1}" for k in range(len(eqs))]
    draw_panel(ax1, A, ROWS, COLS,
               "Column's empirical mix y over row's best-response regions",
               [(n, y) for n, (x, y) in zip(names, eqs)],
               corners=corners, axis_labels=axis_labels[0])
    draw_panel(ax2, B.T, COLS, ROWS,
               "Row's empirical mix x over column's best-response regions",
               [(n, x) for n, (x, y) in zip(names, eqs)],
               corners=corners, axis_labels=axis_labels[1])

    for (name, init), color in zip(starts.items(), LINE_COLORS):
        xs, ys = trajectory(init, corners=corners)
        for ax, pts in ((ax1, ys), (ax2, xs)):
            ax.plot(*pts.T, color=color, lw=1.6, zorder=6, label=name,
                    path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
            ax.plot(*pts[0], "o", color=color, ms=6, mec="white", zorder=7)     # start
            ax.plot(*pts[-1], "X", color=color, ms=9, mec="white", zorder=7)    # end

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=5,
               frameon=False, fontsize=10)
    fig.text(0.5, 0.005, f"o = start (prior counts),  X = after {ITER} rounds",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95), w_pad=5)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    here = Path(__file__).parent
    make_plot(CORNERS, here / "fp_trajectories.png")
    make_plot(RIGHT, here / "fp_trajectories_right.png",
              axis_labels=(("p = P(L)", "q = P(C)"), ("i = P(T)", "j = P(M)")))
    plt.show()
