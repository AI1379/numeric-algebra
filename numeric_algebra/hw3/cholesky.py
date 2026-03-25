import numpy as np
import sympy as sp


def cholesky_np(mat: np.ndarray) -> np.ndarray:
    A = mat.copy()
    n = A.shape[0]
    L = np.zeros_like(A)
    for k in range(n):
        L[k, k] = np.sqrt(A[k, k])
        L[k + 1 : n, k] = A[k + 1 : n, k] / L[k, k]
        for j in range(k + 1, n):
            A[j:n, j] = A[j:n, j] - L[j:n, k] * L[j, k]
    return L


def cholesky_sp(mat: sp.Matrix) -> sp.Matrix:
    A = mat.copy()
    n = A.shape[0]
    L = sp.zeros(n)
    for k in range(n):
        L[k, k] = sp.sqrt(A[k, k])
        L[k + 1 : n, k] = A[k + 1 : n, k] / L[k, k]
        for j in range(k + 1, n):
            A[j:n, j] = A[j:n, j] - L[j:n, k] * L[j, k]
    return L


def cholesky_np_im(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = mat.copy()
    n = A.shape[0]
    D = np.zeros_like(A)
    L = np.eye(n)
    for j in range(n):
        v = np.ndarray(shape=(j,))
        for i in range(j):
            v[i] = L[j, i] * D[i, i]
        D[j, j] = A[j, j] - L[j, :j] @ v
        L[j + 1 : n, j] = (A[j + 1 : n, j] - L[j + 1 : n, :j] @ v) / D[j, j]

    return L, D


def cholesky_sp_im(mat: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    A = mat.copy()
    n = A.shape[0]
    D = sp.zeros(n)
    L = sp.eye(n)
    for j in range(n):
        v = sp.zeros(j, 1)
        for i in range(j):
            v[i, 0] = L[j, i] * D[i, i]
        D[j, j] = A[j, j] - (L[j, :j] * v)[0, 0]
        L[j + 1 : n, j] = (A[j + 1 : n, j] - L[j + 1 : n, :j] * v) / D[j, j]

    return L, D


if __name__ == "__main__":
    A = np.array(
        [
            [4, -2, 4, 2],
            [-2, 10, -2, -7],
            [4, -2, 8, 4],
            [2, -7, 4, 7],
        ],
        dtype=float,
    )
    L = cholesky_np(A)
    print("L:\n", L)
    print("L @ L.T:\n", L @ L.T)
    diff = A - L @ L.T
    print("Difference (A - L @ L.T):\n", diff)

    print("=" * 50)

    L, D = cholesky_np_im(A)
    print("L (improved):\n", L)
    print("D (improved):\n", D)
    print("L @ D @ L.T (improved):\n", L @ D @ L.T)
    diff = A - L @ D @ L.T
    print("Difference (A - L @ D @ L.T):\n", diff)
