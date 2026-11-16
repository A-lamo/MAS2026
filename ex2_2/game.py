import nashpy as nash
import numpy as np

# Rows: T, M, B   Columns: L, C, R
A = np.array([[1, 7, 1],
              [1, 4, 3],
              [3, 1, 4]])   # row player
B = np.array([[6, 8, 12],
              [1, 2, 1],
              [5, 1, 1]])   # column player

game = nash.Game(A, B)


equilibria = game.vertex_enumeration()
for eq in equilibria:
    print("Equilibrium strategies for row and col player: ")
    print(eq)
