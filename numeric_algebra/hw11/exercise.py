import numpy as np


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=500000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x_old = x.copy()
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if np.max(np.abs(x - x_old)) < tol:
            return x, k + 1
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x, max_iter


def jacobi(A, b, tol=1e-10, max_iter=500000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iter):
        x_new = (b - A @ x + np.diag(A) * x) / np.diag(A)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter


def gauss_seidel(A, b, tol=1e-10, max_iter=500000):
    n = len(b)
    x = np.zeros(n)
    D = np.diag(A)
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1 :] @ x_old[i + 1 :]) / D[i]
        if np.max(np.abs(x - x_old)) < tol:
            return x, k + 1
    return x, max_iter


# --- Exercise 2: Hilbert matrix CG test ---


def exercise2():
    print("=" * 60)
    print("Exercise 2: Hilbert matrix CG test")
    print("=" * 60)
    print(f"{'n':>5s}  {'cond(H)':>12s}  {'iters':>7s}  {'||x - x*||_inf':>14s}")
    for n in [3, 5, 8, 10, 15, 20]:
        H = np.array([[1.0 / (i + j + 1) for j in range(1, n + 1)] for i in range(1, n + 1)])
        x_exact = np.full(n, 1.0 / 3)
        b = H @ x_exact
        x_cg, iters = conjugate_gradient(H, b)
        err = np.max(np.abs(x_cg - x_exact))
        cond = np.linalg.cond(H)
        print(f"{n:5d}  {cond:12.4e}  {iters:7d}  {err:14.4e}")


# --- Exercise 3: 5x5 system with Jacobi, G-S, CG ---


def exercise3():
    print()
    print("=" * 60)
    print("Exercise 3: Solve 5x5 system with Jacobi, G-S, CG")
    print("=" * 60)

    A = np.array(
        [
            [0.3, 0.1, 0, 0, 0.2],
            [0.1, 0.3, 0.1, 0, 0],
            [0, 0.1, 0.3, 0.1, 0],
            [0, 0, 0.1, 0.3, 0.1],
            [0.2, 0, 0, 0.1, 0.3],
        ]
    )
    b = np.array([0.6, 0.5, 0.5, 0.5, 0.6])

    x_ref = np.linalg.solve(A, b)

    x_j, it_j = jacobi(A, b)
    x_gs, it_gs = gauss_seidel(A, b)
    x_cg, it_cg = conjugate_gradient(A, b)

    print(f"\nReference (numpy): x = {x_ref}")
    print(f"Jacobi:         {it_j:7d} iters, x = {x_j}, ||x - x_ref|| = {np.max(np.abs(x_j - x_ref)):.4e}")
    print(f"Gauss-Seidel:   {it_gs:7d} iters, x = {x_gs}, ||x - x_ref|| = {np.max(np.abs(x_gs - x_ref)):.4e}")
    print(f"CG:             {it_cg:7d} iters, x = {x_cg}, ||x - x_ref|| = {np.max(np.abs(x_cg - x_ref)):.4e}")


if __name__ == "__main__":
    exercise2()
    exercise3()
