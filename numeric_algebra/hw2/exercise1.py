from typing import Callable
import time

import numpy as np
import scipy as sp


def make_matrix(
    n: int,
    m: int,
    func: Callable[[int, int], float],
    **kwargs,
) -> np.ndarray:
    return np.array(
        [[func(i, j) for j in range(m)] for i in range(n)],
        **kwargs,
    )


def forward_substitution(
    L: np.ndarray, b: np.ndarray, validate: bool = True
) -> np.ndarray:
    """
    Perform forward substitution to solve the system Ly = b.

    Parameters:
    L (np.ndarray): A lower triangular matrix.
    b (np.ndarray): The right-hand side vector.
    validate (bool): If True, validate that L is lower triangular and dimensions are compatible.
    Returns:
    np.ndarray: The solution vector y.
    """
    if validate:
        if not np.allclose(L, np.tril(L)):
            raise ValueError("Matrix L is not a lower triangular matrix.")
        if L.shape[0] != b.shape[0]:
            raise ValueError("Incompatible matrix and vector dimensions.")

    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def backward_substitution(
    U: np.ndarray, y: np.ndarray, validate: bool = True
) -> np.ndarray:
    """
    Perform backward substitution to solve the system Ux = y.

    Parameters:
    U (np.ndarray): An upper triangular matrix.
    y (np.ndarray): The right-hand side vector.
    validate (bool): If True, validate that U is upper triangular and dimensions are compatible.
    Returns:
    np.ndarray: The solution vector x.
    """
    if validate:
        if not np.allclose(U, np.triu(U)):
            raise ValueError("Matrix U is not an upper triangular matrix.")
        if U.shape[0] != y.shape[0]:
            raise ValueError("Incompatible matrix and vector dimensions.")

    n = U.shape[0]
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Zero pivot encountered.")
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


def gaussian_no_pivoting(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Gaussian elimination without pivoting on the matrix A.
    This function transforms the matrix A into an upper triangular matrix U and
    a lower triangular matrix L such that A = LU.

    Parameters:
    A (np.ndarray): The input matrix to be transformed.

    Returns:
    tuple[np.ndarray, np.ndarray]: L and U
    """
    L, U = np.zeros_like(A, dtype=float), A.copy()
    n = A.shape[0]
    for i in range(n):
        L[i, i] = 1.0
        if U[i, i] == 0:
            raise ValueError("Zero pivot encountered.")
        factors = U[i + 1 :, i] / U[i, i]
        L[i + 1 :, i] = factors
        U[i + 1 :] -= np.outer(factors, U[i])
    return L, U


def gaussian_column_pivoting(
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Gaussian elimination with column pivoting on the matrix A.
    This function transforms the matrix A into an upper triangular matrix U and
    a lower triangular matrix L such that PA = LU, where P is a permutation matrix.

    Parameters:
    A (np.ndarray): The input matrix to be transformed.

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: P, L and U
    """
    n = A.shape[0]
    P = np.eye(n)
    L, U = np.zeros_like(A, dtype=float), A.copy()

    for i in range(n):
        # Find the index of the maximum element in the current column
        max_index = np.argmax(np.abs(U[i:, i])) + i
        if U[max_index, i] == 0:
            raise ValueError("Zero pivot encountered.")

        # Swap rows in U and P
        U[[i, max_index]] = U[[max_index, i]]
        P[[i, max_index]] = P[[max_index, i]]

        # Swap rows in L (only the part before the current column)
        if i > 0:
            L[[i, max_index], :i] = L[[max_index, i], :i]

        L[i, i] = 1.0

        factors = U[i + 1 :, i] / U[i, i]
        L[i + 1 :, i] = factors
        U[i + 1 :] -= np.outer(factors, U[i])

    return P, L, U


def solve_by_LU(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x


def main():
    def func(i: int, j: int) -> float:
        if i == j:
            return 6
        elif j == i + 1:
            return 1
        elif j == i - 1:
            return 8
        else:
            return 0

    A = make_matrix(84, 84, func, dtype=float)
    b = np.array(
        [7 if i == 0 else (14 if i == 83 else 15) for i in range(84)], dtype=float
    )

    # Gaussian elimination without pivoting
    time_start = time.perf_counter()
    L_no_pivot, U_no_pivot = gaussian_no_pivoting(A)
    x_no_pivot = solve_by_LU(L_no_pivot, U_no_pivot, b)
    time_end = time.perf_counter()
    print(f"Time taken without pivoting: {time_end - time_start:.6f} seconds")

    # Gaussian elimination with column pivoting
    time_start = time.perf_counter()
    P, L_pivot, U_pivot = gaussian_column_pivoting(A)
    x_pivot = solve_by_LU(L_pivot, U_pivot, P @ b)
    time_end = time.perf_counter()
    print(f"Time taken with column pivoting: {time_end - time_start:.6f} seconds")

    # Solve using SciPy's built-in function for comparison
    time_start = time.perf_counter()
    x_scipy = sp.linalg.solve(A, b)
    time_end = time.perf_counter()
    print(f"Time taken using SciPy function: {time_end - time_start:.6f} seconds")

    diff_no_pivot = np.linalg.norm(A @ x_no_pivot - b)
    diff_pivot = np.linalg.norm(A @ x_pivot - b)
    diff_scipy = np.linalg.norm(A @ x_scipy - b)
    print("Difference for no pivoting:", diff_no_pivot)
    print("Difference for column pivoting:", diff_pivot)
    print("Difference for SciPy's solution:", diff_scipy)
    
    print("Solution without pivoting:", x_no_pivot)
    print("Solution with column pivoting:", x_pivot)
    print("Solution using SciPy:", x_scipy)


if __name__ == "__main__":
    main()
