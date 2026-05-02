import numpy as np
from data import table_3_2_t, table_3_2_y, table_3_3_y, table_3_4_A


def householder_qr(A):
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)
    for k in range(min(m, n)):
        x = R[k:, k].copy()
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            continue
        s = 1.0 if x[0] >= 0 else -1.0
        alpha = -s * norm_x
        v = x.copy()
        v[0] -= alpha
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15:
            continue
        v /= norm_v
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return Q, R


def back_substitution(R, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - R[i, i + 1 :] @ x[i + 1 :]) / R[i, i]
    return x


def qr_solve(A, b):
    m, n = A.shape
    Q, R = householder_qr(A)
    Qtb = Q.T @ b
    if m >= n:
        return back_substitution(R[:n, :n], Qtb[:n])
    raise ValueError("Underdetermined system not supported.")


# --- Task 1: three linear systems from Ch.1 exercises ---


def make_system1():
    # tridiagonal, n=84: diag=6, subdiag=8, superdiag=1
    n = 84
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 6.0
        if i > 0:
            A[i, i - 1] = 8.0
        if i < n - 1:
            A[i, i + 1] = 1.0
    b = np.full(n, 15.0)
    b[0] = 7.0
    b[-1] = 14.0
    return A, b


def make_system2():
    # n=100: diag=10, both sub/superdiag=1
    n = 100
    A = np.eye(n) * 10 + np.eye(n, k=1) + np.eye(n, k=-1)
    b = np.random.rand(n)
    return A, b


def make_system3():
    # n=40 Hilbert: a_ij = 1/(i+j+1), b_i = sum_j 1/(i+j+1)
    n = 40
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    b = np.array([sum(1 / (i + j + 1) for j in range(n)) for i in range(n)])
    return H, b


def task1():
    print("=== Task (1): Solve three linear systems ===\n")

    A1, b1 = make_system1()
    x1_qr = qr_solve(A1, b1)
    x1_np = np.linalg.solve(A1, b1)
    print(f"System 1: tridiagonal, n=84, cond(A) = {np.linalg.cond(A1):.4e}")
    print(f"QR residual ||Ax - b|| = {np.linalg.norm(A1 @ x1_qr - b1):.4e}")
    print(f"NumPy residual ||Ax - b|| = {np.linalg.norm(A1 @ x1_np - b1):.4e}")
    print(f"diff ||x_qr - x_np|| = {np.linalg.norm(x1_qr - x1_np):.4e}")
    print()

    A2, b2 = make_system2()
    x2_qr = qr_solve(A2, b2)
    x2_np = np.linalg.solve(A2, b2)
    print(f"System 2: tridiagonal(10,1,1), n=100, cond(A) = {np.linalg.cond(A2):.4e}")
    print(f"QR residual ||Ax - b|| = {np.linalg.norm(A2 @ x2_qr - b2):.4e}")
    print(f"NumPy residual ||Ax - b|| = {np.linalg.norm(A2 @ x2_np - b2):.4e}")
    print(f"diff ||x_qr - x_np|| = {np.linalg.norm(x2_qr - x2_np):.4e}")
    print()

    A3, b3 = make_system3()
    x3_qr = qr_solve(A3, b3)
    x3_np = np.linalg.solve(A3, b3)
    print(f"System 3: Hilbert, n=40, cond(A) = {np.linalg.cond(A3):.4e}")
    print(f"QR residual ||Ax - b|| = {np.linalg.norm(A3 @ x3_qr - b3):.4e}")
    print(f"NumPy residual ||Ax - b|| = {np.linalg.norm(A3 @ x3_np - b3):.4e}")
    print(f"diff ||x_qr - x_np|| = {np.linalg.norm(x3_qr - x3_np):.4e}")


# --- Task 2: quadratic polynomial fitting ---


def task2():
    print("\n=== Task (2): Quadratic polynomial fitting ===\n")

    t, y = table_3_2_t, table_3_2_y
    A = np.column_stack([t**2, t, np.ones_like(t)])
    coeff = qr_solve(A, y)
    a, b, c = coeff
    residual = np.linalg.norm(A @ coeff - y)
    print(f"y = {a:.6f} t^2 + {b:.6f} t + {c:.6f}")
    print(f"residual ||r||_2 = {residual:.6e}")
    coeff_np, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    print(f"NumPy lstsq diff = {np.linalg.norm(coeff - coeff_np):.4e}")


# --- Task 3: house price linear model ---


def task3():
    print("\n=== Task (3): House price linear model ===\n")

    y = table_3_3_y
    m = table_3_4_A.shape[0]
    A = np.column_stack([np.ones(m), table_3_4_A])
    coeff = qr_solve(A, y)
    residual = np.linalg.norm(A @ coeff - y)
    labels = [
        "x0",
        "a1 (tax)",
        "a2 (bathrooms)",
        "a3 (lot size)",
        "a4 (living area)",
        "a5 (garage)",
        "a6 (rooms)",
        "a7 (bedrooms)",
        "a8 (age)",
        "a9 (build type)",
        "a10 (layout)",
        "a11 (fireplace)",
    ]
    print("Least squares coefficients:")
    for label, val in zip(labels, coeff):
        print(f"  {label:20s} = {val:.6f}")
    print(f"\nresidual ||r||_2 = {residual:.6e}")
    coeff_np, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    print(f"NumPy lstsq diff = {np.linalg.norm(coeff - coeff_np):.4e}")


if __name__ == "__main__":
    task1()
    task2()
    task3()
