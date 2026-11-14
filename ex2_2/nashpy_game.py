"""Exercise 2.2 - NashPy: equilibria of a 3x3 bimatrix game."""
from fractions import Fraction

import nashpy as nash
import numpy as np

# Rows: T, M, B   Columns: L, C, R
A = np.array([[1, 7, 1],
              [1, 4, 3],
              [3, 1, 4]])   # row player
B = np.array([[6, 8, 12],
              [1, 2, 1],
              [5, 1, 1]])   # column player
ROWS, COLS = "TMB", "LCR"
game = nash.Game(A, B)


def fmt(v):
    return "(" + ", ".join(str(Fraction(x).limit_denominator(1000)) for x in v) + ")"


def show(sigma_r, sigma_c, label=""):
    u_r, u_c = game[sigma_r, sigma_c]
    print(f"{label} row={fmt(sigma_r)}  col={fmt(sigma_c)}  "
          f"payoffs=({Fraction(u_r).limit_denominator(1000)}, "
          f"{Fraction(u_c).limit_denominator(1000)})")


# ---------------------------------------------------------------- Q1
print("=== Q1: support enumeration ===")
equilibria = list(game.support_enumeration())
for eq in equilibria:
    show(*eq)

print("\n=== Q1 cross-check: vertex enumeration ===")
for eq in game.vertex_enumeration():
    show(*eq)

# ---------------------------------------------------------------- Q2
print("\n=== Q2: mixed NE with reduced (2x2) support, solved via indifference ===")
# Generic 2x2 solver for supports (i,j) x (k,l):
def solve_2x2(ri, ci):
    (i, j), (k, l) = ri, ci
    # column prob q on k makes row indifferent between i and j
    # A[i,k]q + A[i,l](1-q) = A[j,k]q + A[j,l](1-q)
    den_q = (A[i, k] - A[i, l]) - (A[j, k] - A[j, l])
    den_p = (B[i, k] - B[j, k]) - (B[i, l] - B[j, l])
    if den_q == 0 or den_p == 0:
        return None
    q = Fraction(int(A[j, l] - A[i, l]), int(den_q))
    p = Fraction(int(B[j, l] - B[j, k]), int(den_p))
    if not (0 < p < 1 and 0 < q < 1):
        return None
    x = np.zeros(3, dtype=object); x[i], x[j] = p, 1 - p
    y = np.zeros(3, dtype=object); y[k], y[l] = q, 1 - q
    # check no profitable deviation to an action outside the support
    Ay = A @ y; xB = x @ B
    if max(Ay) > Ay[i] or max(xB) > xB[k]:
        return None
    return x, y

from itertools import combinations
for ri in combinations(range(3), 2):
    for ci in combinations(range(3), 2):
        sol = solve_2x2(ri, ci)
        if sol:
            x, y = sol
            print(f"support {ROWS[ri[0]]}{ROWS[ri[1]]} x {COLS[ci[0]]}{COLS[ci[1]]}: "
                  f"row={fmt(x)} col={fmt(y)}  "
                  f"payoffs=({x @ A @ y}, {x @ B @ y})")

# ---------------------------------------------------------------- Q3
print("\n=== Q3: fictitious play ===")
rng = np.random.default_rng(0)
ITER = 10_000


def fp_run(init_r, init_c, iterations=ITER):
    """Fictitious play from given initial play counts; return empirical mixtures."""
    counts = list(game.fictitious_play(iterations=iterations,
                                       play_counts=(np.array(init_r, float),
                                                    np.array(init_c, float))))
    r, c = counts[-1]
    return r / r.sum(), c / c.sum()


starts = {"default (no prior)": None}
for name, (r, c) in {"T,C": ((1, 0, 0), (0, 1, 0)),
                     "M,R": ((0, 1, 0), (0, 0, 1)),
                     "B,L": ((0, 0, 1), (1, 0, 0)),
                     "T,R": ((1, 0, 0), (0, 0, 1)),
                     "uniform": ((1, 1, 1), (1, 1, 1))}.items():
    starts[name] = (r, c)
for k in range(10):
    starts[f"random #{k}"] = (rng.integers(0, 20, 3), rng.integers(0, 20, 3))

found = {}
for name, init in starts.items():
    if init is None:
        r, c = list(game.fictitious_play(iterations=ITER))[-1]
        x, y = r / r.sum(), c / c.sum()
    else:
        x, y = fp_run(*init)
    key = (tuple(np.round(x, 1).tolist()), tuple(np.round(y, 1).tolist()))
    found.setdefault(key, []).append(name)
    print(f"{name:20s} -> row={np.round(x, 3)} col={np.round(y, 3)}")

print("\nDistinct limits reached by fictitious play:")
for (x, y), names in found.items():
    print(f"  row={x} col={y}   <- {len(names)} start(s)")

# Start FP *exactly at* each mixed NE (large prior counts proportional to it):
print("\nFP started at the mixed equilibria (prior weight 1000):")
for x0, y0 in equilibria[1:]:
    x, y = fp_run(np.round(1000 * x0), np.round(1000 * y0), iterations=20_000)
    print(f"  start row={fmt(x0)} col={fmt(y0)} -> row={np.round(x, 3)} col={np.round(y, 3)}")
