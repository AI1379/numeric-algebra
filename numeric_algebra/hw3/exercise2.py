import time
import sys

import numpy as np

from numeric_algebra.common import (
    backward_substitution,
    forward_substitution,
    gaussian_column_pivoting,
    gaussian_no_pivoting,
    solve_by_LU,
)
from numeric_algebra.hw3.cholesky import cholesky_np, cholesky_np_im


def solve_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    L = cholesky_np(A)
    y = forward_substitution(L, b, validate=False)
    x = backward_substitution(L.T, y, validate=False)
    return x


def solve_cholesky_im(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    L, D = cholesky_np_im(A)
    y = forward_substitution(L, b, validate=False)
    DLT = D @ L.T
    x = backward_substitution(DLT, y, validate=False)
    return x


def get_hilbert_matrix(n: int) -> np.ndarray:
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    return H


def get_hilbert_vector(n: int) -> np.ndarray:
    b = np.zeros(n)
    for i in range(n):
        b[i] = sum(1 / (i + j + 1) for j in range(n))
    return b


def get_bar_matrix(n: int) -> np.ndarray:
    return np.eye(n) * 10 + np.eye(n, k=1) + np.eye(n, k=-1)


def all_four_solutions(A: np.ndarray, b: np.ndarray):
    t1 = time.perf_counter()
    L, U = gaussian_no_pivoting(A)
    x_no_pivot = solve_by_LU(L, U, b)
    t2 = time.perf_counter()
    t_no_pivot = t2 - t1

    t1 = time.perf_counter()
    P, L_col, U_col = gaussian_column_pivoting(A)
    x_col_pivot = solve_by_LU(L_col, U_col, P @ b)
    t2 = time.perf_counter()
    t_col_pivot = t2 - t1

    t1 = time.perf_counter()
    x_cholesky = solve_cholesky(A, b)
    t2 = time.perf_counter()
    t_cholesky = t2 - t1

    t1 = time.perf_counter()
    x_cholesky_im = solve_cholesky_im(A, b)
    t2 = time.perf_counter()
    t_cholesky_im = t2 - t1

    diff_no_pivot = np.linalg.norm(A @ x_no_pivot - b)
    diff_col_pivot = np.linalg.norm(A @ x_col_pivot - b)
    diff_cholesky = np.linalg.norm(A @ x_cholesky - b)
    diff_cholesky_im = np.linalg.norm(A @ x_cholesky_im - b)

    print("Solution with Cholesky:\n", x_cholesky)
    print("Solution with Cholesky IM:\n", x_cholesky_im)

    print(
        f"Difference for no pivoting: {diff_no_pivot}, time: {t_no_pivot:.4f} seconds"
    )
    print(
        f"Difference for column pivoting: {diff_col_pivot}, time: {t_col_pivot:.4f} seconds"
    )
    print(f"Difference for Cholesky: {diff_cholesky}, time: {t_cholesky:.4f} seconds")
    print(
        f"Difference for Cholesky IM: {diff_cholesky_im}, time: {t_cholesky_im:.4f} seconds"
    )


def gen_symmetric_pos_def_matrix(n: int) -> np.ndarray:
    A = np.abs(np.random.rand(n, n))
    return A + A.T


def perf_test():
    TEST_SIZES = [100, 200, 500, 1000]
    TEST_ROUNDS = 10
    for n in TEST_SIZES:
        cholesky_im_tot = 0
        gaussian_col_tot = 0
        print(f"\nTesting performance for size {n}...", file=sys.stderr)

        for _ in range(TEST_ROUNDS):
            A = gen_symmetric_pos_def_matrix(n)
            b = np.random.rand(n)

            start = time.perf_counter()
            solve_cholesky_im(A, b)
            cholesky_im_tot += time.perf_counter() - start

            start = time.perf_counter()
            P, L_col, U_col = gaussian_column_pivoting(A)
            solve_by_LU(L_col, U_col, P @ b)
            gaussian_col_tot += time.perf_counter() - start

        print(
            f"Size: {n}, Cholesky IM avg: {cholesky_im_tot / TEST_ROUNDS:.4f} seconds, "
            f"Gaussian Col Pivoting avg: {gaussian_col_tot / TEST_ROUNDS:.4f} seconds"
        )


def main():
    CMAT = get_bar_matrix(100)
    b = np.random.rand(100)
    all_four_solutions(CMAT, b)

    H = get_hilbert_matrix(40)
    b_hilbert = get_hilbert_vector(40)
    all_four_solutions(H, b_hilbert)

    perf_test()


if __name__ == "__main__":
    main()
