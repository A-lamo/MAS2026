"""Exercise 2.2 - fictitious play from batches of players started at each corner.

Each player gets 3 batches, one per pure strategy (row: T, M, B; column: L, C, R).
A player in a batch starts with prior counts near that corner:
    prior = W * ((1 - EPS) * corner + EPS * random point on the simplex)
Every row batch is paired with every column batch (3 x 3 = 9 combinations).
Prints where each combination ends up and plots the runs on the right triangles,
each player's runs coloured by that player's starting corner.
(fp_corner_batches_gif.py animates the same runs.)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np

from best_response_plot import A, B, COLS, ROWS, draw_panel
from fp_trajectory_plot import RIGHT

N, W, EPS, ITER = 15, 5, 0.3, 3000   # players per batch, prior weight, jitter, rounds
BATCH_COLORS = ["#C62828", "#6A1B9A", "#111111"]

game = nash.Game(A, B)
eqs = list(game.support_enumeration())


def nearest_ne(x, y, tol=0.1):
    dist = [np.abs(x - a).sum() + np.abs(y - b).sum() for a, b in eqs]
    k = int(np.argmin(dist))
    return f"NE{k + 1}" if dist[k] < tol else "not converged"


def run(init):
    xs, ys = [], []
    for r, c in game.fictitious_play(iterations=ITER, play_counts=init):
        xs.append(r / r.sum()); ys.append(c / c.sum())
    return np.array(xs), np.array(ys)


def batch_runs():
    """All runs as (row corner, col corner, xs, ys); xs/ys are empirical mixes per round."""
    np.random.seed(0)                # nashpy breaks best-response ties with np.random
    rng = np.random.default_rng(0)

    def prior(corner):
        return W * ((1 - EPS) * np.eye(3)[corner] + EPS * rng.dirichlet(np.ones(3)))

    return [(ri, ci, *run((prior(ri), prior(ci))))
            for ri in range(3) for ci in range(3) for _ in range(N)]


def cluster_runs(k, n=40, w=30, eps=0.1, seed=0):
    """n runs starting near equilibrium eqs[k]: prior = w * ((1-eps) * NE + eps * random)."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    return [run(tuple(w * ((1 - eps) * v + eps * rng.dirichlet(np.ones(3))) for v in eqs[k]))
            for _ in range(n)]


def setup_figure(footer, titles=("Column's runs, coloured by column's start corner",
                                 "Row's runs, coloured by row's start corner"),
                 legends=None, bottom=0.05):
    """Right-triangle panels with best-response regions, legends and footer text.
    legends: [(label, color), ...] shown in both panels; default = start corners."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.8))
    names = [f"NE{k + 1}" for k in range(len(eqs))]
    draw_panel(ax1, A, ROWS, COLS, titles[0],
               [(n, y) for n, (x, y) in zip(names, eqs)],
               corners=RIGHT, axis_labels=("p = P(L)", "q = P(C)"))
    draw_panel(ax2, B.T, COLS, ROWS, titles[1],
               [(n, x) for n, (x, y) in zip(names, eqs)],
               corners=RIGHT, axis_labels=("i = P(T)", "j = P(M)"))
    for ax, labels in ((ax1, COLS), (ax2, ROWS)):
        entries = legends or [(f"started near {lab}", BATCH_COLORS[k])
                              for k, lab in enumerate(labels)]
        for label, color in entries:
            ax.plot([], [], color=color, lw=2, label=label)
        ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.text(0.5, 0.012, footer, ha="center", fontsize=10, color="dimgray")
    fig.tight_layout(rect=(0, bottom, 1, 0.94), w_pad=5)
    return fig, ax1, ax2


if __name__ == "__main__":
    runs = batch_runs()

    print(f"{N} players per batch, prior weight {W}, jitter {EPS}, {ITER} rounds\n")
    print("row start \\ col start   " + "".join(f"{c:<22s}" for c in COLS))
    for ri in range(3):
        cells = []
        for ci in range(3):
            hits = {}
            for r, c, xs, ys in runs:
                if (r, c) == (ri, ci):
                    name = nearest_ne(xs[-1], ys[-1])
                    hits[name] = hits.get(name, 0) + 1
            cells.append(", ".join(f"{k} {v}" for k, v in sorted(hits.items())))
        print(f"{ROWS[ri]:<24s}" + "".join(f"{c:<22s}" for c in cells))

    fig, ax1, ax2 = setup_figure(
        f"{N} players per batch x 9 batch pairs;  o = start,  X = after {ITER} rounds")
    for ri, ci, xs, ys in runs:
        for ax, pts, k in ((ax1, ys @ RIGHT, ci), (ax2, xs @ RIGHT, ri)):
            ax.plot(*pts.T, color=BATCH_COLORS[k], lw=0.8, alpha=0.35, zorder=6)
            ax.plot(*pts[0], "o", color=BATCH_COLORS[k], ms=3, zorder=7)
            ax.plot(*pts[-1], "X", color=BATCH_COLORS[k], ms=7, mec="white", zorder=7)

    out = Path(__file__).with_name("fp_corner_batches.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")
    plt.show()
