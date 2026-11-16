"""Exercise 2.2 - 3x3 analogue of the 2x2 "payoff vs. opponent's probability" plot.

In a 2x2 game each pure strategy's expected payoff is a LINE over the opponent's
probability. In a 3x3 game it is a PLANE over the opponent's (p, q) triangle:

    row player,    y = (p, q, r = 1-p-q):  T = 1 + 6q,  M = 3 - 2p + q,  B = 4 - p - 3q
    column player, x = (i, j, k = 1-i-j):  L = 5 + i - 4j,  C = 1 + 7i + j,  R = 1 + 11i

The best response is the upper envelope (max) of the three planes. Projecting
that envelope down onto the triangle gives the best-response regions, and its
creases project onto the indifference lines (T = M, etc.).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from best_response_plot import A, B, COLORS, COLS, ROWS, br_regions


def draw(ax, M, own, axis_names, title, ne_points):
    """M: own payoffs (own strategies x opponent strategies).
    The opponent's mix is (a, b, 1-a-b); we plot over the (a, b) right triangle."""
    tri = np.eye(3)                                  # barycentric corners
    z_floor = M.min() - 1.5

    for s in range(3):
        # full plane over the whole triangle: faint
        corners = [(*t[:2], M[s] @ t) for t in tri]
        ax.add_collection3d(Poly3DCollection([corners], alpha=0.12,
                                             facecolor=COLORS[s], edgecolor=COLORS[s], zorder=2))
    for s, poly in enumerate(br_regions(M)):
        if len(poly) < 3:
            continue
        # the part of plane s that is the upper envelope: solid
        top = [(*p[:2], M[s] @ p) for p in poly]
        ax.add_collection3d(Poly3DCollection([top], alpha=0.9, facecolor=COLORS[s],
                                             edgecolor="black", linewidth=1.2, zorder=3))
        # its shadow on the floor = best-response region
        floor = [(*p[:2], z_floor) for p in poly]
        ax.add_collection3d(Poly3DCollection([floor], alpha=0.6, facecolor=COLORS[s],
                                             edgecolor="black", linewidth=0.8, zorder=1))
        c = np.mean(top, axis=0)
        ax.text(c[0], c[1], c[2] + 0.4, own[s], fontsize=14, fontweight="bold", zorder=4)

    for label, p, marker in ne_points:
        p = np.asarray(p, float)
        z = (M @ p).max()
        ax.plot([p[0], p[0]], [p[1], p[1]], [z_floor, z], "k:", lw=1, zorder=9)
        ax.scatter(p[0], p[1], z, marker=marker, s=90, c="yellow",
                   edgecolors="black", depthshade=False, zorder=10, label=label)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(z_floor, M.max() + 0.5)
    ax.set_xlabel(axis_names[0]); ax.set_ylabel(axis_names[1])
    ax.set_zlabel("expected payoff")
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=22, azim=-125)


if __name__ == "__main__":
    import nashpy as nash
    eqs = list(nash.Game(A, B).support_enumeration())
    markers = ["o", "s", "*"]

    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    draw(ax1, A, ROWS, ("p = P(L)", "q = P(C)"),
         "Row player: payoff of T, M, B over column mix (p, q)",
         [(f"NE{n+1}", y, m) for n, ((x, y), m) in enumerate(zip(eqs, markers))])
    ax2 = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    draw(ax2, B.T, COLS, ("i = P(T)", "j = P(M)"),
         "Column player: payoff of L, C, R over row mix (i, j)",
         [(f"NE{n+1}", x, m) for n, ((x, y), m) in enumerate(zip(eqs, markers))])
    ax2.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out = Path(__file__).with_name("payoff_planes.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.show()
