"""Exercise 2.2 - NashPy: equilibria of a 3x3 bimatrix game."""
from fractions import Fraction
from itertools import combinations

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
    print(f"{label} row={fmt(sigma_r)}  col={fmt(sigma_c)}  "
          f"payoffs={fmt(game[sigma_r, sigma_c])}")


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


def indifference(M, own, opp):
    """Probability w the opponent puts on opp[0] (and 1-w on opp[1]) that makes a
    player with payoffs M (own actions x opponent actions) indifferent between
    own[0] and own[1]:  d[opp[0]] w + d[opp[1]] (1-w) = 0,  d = M[own[0]] - M[own[1]]."""
    d = M[own[0]] - M[own[1]]
    dk, dl = int(d[opp[0]]), int(d[opp[1]])
    return None if dk == dl else Fraction(dl, dl - dk)


def solve_2x2(ri, ci):
    """Mixed NE with row support ri = (i, j) and column support ci = (k, l), or None."""
    q = indifference(A, ri, ci)     # column's prob on k: makes row indifferent
    p = indifference(B.T, ci, ri)   # row's prob on i: makes column indifferent
    if p is None or q is None or not (0 < p < 1 and 0 < q < 1):
        return None
    x = np.zeros(3, dtype=object); x[list(ri)] = p, 1 - p
    y = np.zeros(3, dtype=object); y[list(ci)] = q, 1 - q
    # no profitable deviation to an action outside the support
    Ay, xB = A @ y, x @ B
    if max(Ay) > Ay[ri[0]] or max(xB) > xB[ci[0]]:
        return None
    return x, y


for ri in combinations(range(3), 2):
    for ci in combinations(range(3), 2):
        if sol := solve_2x2(ri, ci):
            x, y = sol
            print(f"support {ROWS[ri[0]]}{ROWS[ri[1]]} x {COLS[ci[0]]}{COLS[ci[1]]}: "
                  f"row={fmt(x)} col={fmt(y)}  "
                  f"payoffs=({x @ A @ y}, {x @ B @ y})")

# ---------------------------------------------------------------- Q3
print("\n=== Q3: fictitious play ===")
np.random.seed(0)             # nashpy breaks best-response ties with np.random
rng = np.random.default_rng(0)
ITER = 10_000


def fp_run(init=None, iterations=ITER):
    """Fictitious play from initial play counts (row, col), or none; return empirical mixtures."""
    counts = None if init is None else tuple(np.array(c, float) for c in init)
    for r, c in game.fictitious_play(iterations=iterations, play_counts=counts):
        pass                                            # keep only the final counts
    return r / r.sum(), c / c.sum()


starts = {"default (no prior)": None,
          "T,C": ((1, 0, 0), (0, 1, 0)),
          "M,R": ((0, 1, 0), (0, 0, 1)),
          "B,L": ((0, 0, 1), (1, 0, 0)),
          "T,R": ((1, 0, 0), (0, 0, 1)),
          "uniform": ((1, 1, 1), (1, 1, 1))}
for k in range(10):
    starts[f"random #{k}"] = (rng.integers(0, 20, 3), rng.integers(0, 20, 3))

found = {}
for name, init in starts.items():
    x, y = fp_run(init)
    key = (tuple(np.round(x, 1).tolist()), tuple(np.round(y, 1).tolist()))
    found.setdefault(key, []).append(name)
    print(f"{name:20s} -> row={np.round(x, 3)} col={np.round(y, 3)}")

print("\nDistinct limits reached by fictitious play:")
for (x, y), names in found.items():
    print(f"  row={x} col={y}   <- {len(names)} start(s)")

# Start FP *exactly at* each mixed NE (large prior counts proportional to it):
print("\nFP started at the mixed equilibria (prior weight 1000):")
for x0, y0 in equilibria[1:]:
    x, y = fp_run((np.round(1000 * x0), np.round(1000 * y0)), iterations=20_000)
    print(f"  start row={fmt(x0)} col={fmt(y0)} -> row={np.round(x, 3)} col={np.round(y, 3)}")


def nearest_ne(x, y, tol=0.1):
    """Name of the equilibrium within L1 distance tol of (x, y), else 'not converged'."""
    dist = [np.abs(x - a).sum() + np.abs(y - b).sum() for a, b in equilibria]
    k = int(np.argmin(dist))
    return f"NE{k + 1}" if dist[k] < tol else "not converged"


# Basins: many random priors (uniform on the simplex, times a prior weight w)
RUNS, BASIN_ITER = 50, 10_000
print(f"\nBasin counts: {RUNS} random priors per weight w, {BASIN_ITER} rounds each")
for w in (1, 10, 100):
    hits = {}
    for _ in range(RUNS):
        init = (w * rng.dirichlet(np.ones(3)), w * rng.dirichlet(np.ones(3)))
        name = nearest_ne(*fp_run(init, iterations=BASIN_ITER))
        hits[name] = hits.get(name, 0) + 1
    print(f"  w={w:<4d} " + "  ".join(f"{k}: {v}" for k, v in sorted(hits.items())))

# Stability: start near each mixed NE, moving a fraction eps of the prior to uniform
print("\nStability: prior 100 at (1-eps)*NE + eps*uniform, 10 runs each")
for k, (x0, y0) in enumerate(equilibria[1:], start=2):
    for eps in (0.05, 0.2):
        hits = {}
        for _ in range(10):
            init = tuple(100 * ((1 - eps) * v + eps / 3) for v in (x0, y0))
            name = nearest_ne(*fp_run(init, iterations=BASIN_ITER))
            hits[name] = hits.get(name, 0) + 1
        print(f"  NE{k}, eps={eps:<4}  -> " + "  ".join(f"{n}: {c}" for n, c in sorted(hits.items())))
