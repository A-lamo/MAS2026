"""Exercise 2.2 - fictitious play from a cluster of starts around NE2 and around NE3.

Each run starts with prior counts  W * ((1 - EPS) * NE + EPS * random point on the
simplex), i.e. near the equilibrium. Runs are coloured by their starting distance
from the equilibrium,  d = |x0 - x*|_1 + |y0 - y*|_1  (same colour in both triangles);
the legend counts where the runs end up.
Saves fp_cluster_NE2.gif and fp_cluster_NE3.gif.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, to_hex
from matplotlib.ticker import MaxNLocator

from fp_corner_batches import cluster_runs, eqs, nearest_ne, setup_figure
from fp_corner_batches_gif import animate

N, W, EPS = 100, 30, 0.2
OUTCOMES = ("NE1", "NE2", "NE3", "not converged")
CMAP = ListedColormap(plt.cm.plasma(np.linspace(0, 0.88, 256)))   # skip pale-yellow end

for k in (1, 2):
    x_star, y_star = eqs[k]
    runs = cluster_runs(k, n=N, w=W, eps=EPS)
    ends = [nearest_ne(xs[-1], ys[-1]) for xs, ys in runs]
    dist = np.array([np.abs(xs[0] - x_star).sum() + np.abs(ys[0] - y_star).sum()
                     for xs, ys in runs])
    counts = {e: ends.count(e) for e in OUTCOMES if e in ends}
    print(f"NE{k + 1} cluster: " + ", ".join(
        f"{e}: {c} (mean start distance {dist[[x == e for x in ends]].mean():.3f})"
        for e, c in counts.items()))

    fig, ax1, ax2 = setup_figure(
        f"{N} players started near NE{k + 1} (prior weight {W}, jitter {EPS})",
        titles=("Column's runs", "Row's runs"),
        legends=[(f"{'ends at ' if e.startswith('NE') else ''}{e}: {c}/{N}", "none")
                 for e, c in counts.items()],
        bottom=0.15)
    norm = Normalize(dist.min(), dist.max())
    cax = fig.add_axes((0.35, 0.085, 0.3, 0.02))
    cbar = fig.colorbar(ScalarMappable(norm, CMAP), cax=cax, orientation="horizontal",
                        label=f"starting distance from NE{k + 1}  (|x0 - x*| + |y0 - y*|)")
    cbar.locator = MaxNLocator(5); cbar.update_ticks()

    order = np.argsort(dist)          # far runs drawn last, so escaping trails stay visible
    tracks = [(runs[i][1], runs[i][0], to_hex(CMAP(norm(dist[i]))),
               to_hex(CMAP(norm(dist[i])))) for i in order]
    animate(fig, ax1, ax2, tracks, Path(__file__).with_name(f"fp_cluster_NE{k + 1}.gif"),
            head_size=40, start_size=0, trail_width=0.4, trail_alpha=0.25, head_edge=0.6)
