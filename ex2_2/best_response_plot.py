"""Exercise 2.2 - best-response regions of the 3x3 game, drawn on the strategy simplices.

In a 3x3 game a mixed strategy is a point in a triangle, so the "best-response
curves" become a partition of the opponent's simplex into regions where each
pure strategy is a best response. The boundaries are the indifference lines;
a Nash equilibrium is a pair (x, y) where y lies in the region(s) of every
strategy in x's support, and vice versa.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np

# Same game as nashpy_game.py.  Rows: T, M, B   Columns: L, C, R
A = np.array([[1, 7, 1],
              [1, 4, 3],
              [3, 1, 4]])   # row player
B = np.array([[6, 8, 12],
              [1, 2, 1],
              [5, 1, 1]])   # column player
ROWS, COLS = "TMB", "LCR"

# Triangle corners in the plane; a mix p = (p0, p1, p2) maps to p @ CORNERS.
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def clip(poly, w):
    """Sutherland-Hodgman: keep the part of poly (barycentric points) where w @ p >= 0."""
    out = []
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        fp, fq = w @ p, w @ q
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            out.append(p + fp / (fp - fq) * (q - p))
    return out


def br_regions(M):
    """For payoff matrix M (own strategies x opponent strategies), return for each
    own pure strategy i the polygon of opponent mixes y where i is a best response."""
    regions = []
    for i in range(3):
        poly = [np.eye(3)[k] for k in range(3)]
        for j in range(3):
            if j != i and poly:
                poly = clip(poly, M[i] - M[j])   # (M y)_i - (M y)_j >= 0
        regions.append(poly)
    return regions


def draw_panel(ax, M, own_labels, opp_labels, title, ne_points):
    for i, poly in enumerate(br_regions(M)):
        if len(poly) < 3:
            continue
        xy = np.array(poly) @ CORNERS
        ax.fill(*xy.T, color=COLORS[i], alpha=0.55, lw=1.5, ec="black")
        cx, cy = xy.mean(axis=0)
        ax.text(cx, cy, own_labels[i], ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
    tri = np.vstack([CORNERS, CORNERS[:1]])
    ax.plot(*tri.T, color="black", lw=1.5)
    offsets = [(-0.06, -0.05), (0.06, -0.05), (0, 0.05)]
    for (x, y), (dx, dy), lab in zip(CORNERS, offsets, opp_labels):
        ax.text(x + dx, y + dy, lab, ha="center", va="center", fontsize=13)
    for (label, p), marker in zip(ne_points, ["o", "s", "*"]):
        px, py = np.asarray(p, float) @ CORNERS
        ax.plot(px, py, marker, ms=13 if marker == "*" else 10,
                mfc="yellow", mec="black", mew=1.5, label=label, zorder=5)
    ax.set_title(title, fontsize=12, pad=25)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    equilibria = list(nash.Game(A, B).support_enumeration())
    names = [f"NE{k + 1}" for k in range(len(equilibria))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    draw_panel(ax1, A, ROWS, COLS,
               "Row player's best response (T/M/B)\nover column's mixes y on {L, C, R}",
               [(n, y) for n, (x, y) in zip(names, equilibria)])
    draw_panel(ax2, B.T, COLS, ROWS,
               "Column player's best response (L/C/R)\nover row's mixes x on {T, M, B}",
               [(n, x) for n, (x, y) in zip(names, equilibria)])
    ax2.legend(loc="upper right", fontsize=11, frameon=False)
    fig.tight_layout(w_pad=5)

    out = Path(__file__).with_name("best_response.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.show()
