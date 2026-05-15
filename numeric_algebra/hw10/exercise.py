import numpy as np


def build_system(eps, a, n):
    h = 1.0 / n
    N = n - 1
    d = -(2 * eps + h) * np.ones(N)
    sub = eps * np.ones(N - 1)
    sup = (eps + h) * np.ones(N - 1)
    b = a * h**2 * np.ones(N)
    b[-1] -= eps + h
    return d, sub, sup, b


def exact_solution(x, eps, a):
    if eps < 1e-300:
        return a * x
    c = (1 - a) / (1 - np.exp(-1.0 / eps))
    return c * (1 - np.exp(-x / eps)) + a * x


def jacobi(d, sub, sup, b, tol=1e-8, max_iter=500000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iter):
        x_new = b.copy()
        x_new[:-1] -= sup * x[1:]
        x_new[1:] -= sub * x[:-1]
        x_new /= d
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter


def gauss_seidel(d, sub, sup, b, tol=1e-8, max_iter=500000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iter):
        x_old = x.copy()
        x[0] = (b[0] - sup[0] * x[1]) / d[0]
        for i in range(1, n - 1):
            x[i] = (b[i] - sub[i - 1] * x[i - 1] - sup[i] * x[i + 1]) / d[i]
        x[-1] = (b[-1] - sub[-1] * x[-2]) / d[-1]
        if np.max(np.abs(x - x_old)) < tol:
            return x, k + 1
    return x, max_iter


def sor(d, sub, sup, b, omega, tol=1e-8, max_iter=500000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iter):
        x_old = x.copy()
        x[0] = (1 - omega) * x[0] + omega * (b[0] - sup[0] * x[1]) / d[0]
        for i in range(1, n - 1):
            x[i] = (1 - omega) * x[i] + omega * (
                b[i] - sub[i - 1] * x[i - 1] - sup[i] * x[i + 1]
            ) / d[i]
        x[-1] = (1 - omega) * x[-1] + omega * (
            b[-1] - sub[-1] * x[-2]
        ) / d[-1]
        if np.max(np.abs(x - x_old)) < tol:
            return x, k + 1
    return x, max_iter


def jacobi_spectral_radius(eps, n):
    """Analytical spectral radius of Jacobi iteration matrix for this tridiagonal system.

    Eigenvalues: lambda_k = 2*sqrt(eps*(eps+h)) / (2*eps+h) * cos(k*pi/n), k=1,...,n-1
    """
    h = 1.0 / n
    return 2 * np.sqrt(eps * (eps + h)) / (2 * eps + h) * np.cos(np.pi / n)


def solve(eps, a=0.5, n=100):
    print(f"=== eps = {eps}, a = {a}, n = {n} ===\n")
    d, sub, sup, b = build_system(eps, a, n)
    h = 1.0 / n
    N = n - 1
    x_grid = np.linspace(h, 1 - h, N)
    y_exact = exact_solution(x_grid, eps, a)

    rho_J = jacobi_spectral_radius(eps, n)
    omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_J**2))
    print(f"rho(B_J) = {rho_J:.6f}, omega_opt = {omega_opt:.6f}")

    # Jacobi
    y_j, it_j = jacobi(d, sub, sup, b)
    err_j = np.max(np.abs(y_j - y_exact))
    print(f"Jacobi:         {it_j:7d} iters, max|y - y*| = {err_j:.4e}")

    # Gauss-Seidel
    y_gs, it_gs = gauss_seidel(d, sub, sup, b)
    err_gs = np.max(np.abs(y_gs - y_exact))
    print(f"Gauss-Seidel:   {it_gs:7d} iters, max|y - y*| = {err_gs:.4e}")

    # SOR
    y_sor, it_sor = sor(d, sub, sup, b, omega_opt)
    err_sor = np.max(np.abs(y_sor - y_exact))
    print(f"SOR(w={omega_opt:.4f}):  {it_sor:7d} iters, max|y - y*| = {err_sor:.4e}")
    print()


def solve_vary_n(eps, a=0.5):
    """For a given eps, solve with different n and show how error scales."""
    print(f"=== eps = {eps}, a = {a}, varying n ===")
    print(f"{'n':>7s}  {'h':>10s}  {'h/eps':>10s}  {'max|y - y*|':>12s}  {'iters(GS)':>10s}")
    for n in [100, 200, 500, 1000, 2000, 5000]:
        d, sub, sup, b = build_system(eps, a, n)
        h = 1.0 / n
        N = n - 1
        x_grid = np.linspace(h, 1 - h, N)
        y_exact = exact_solution(x_grid, eps, a)

        # optimal SOR
        rho_J = jacobi_spectral_radius(eps, n)
        omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_J**2))

        y_sor, it_sor = sor(d, sub, sup, b, omega_opt)
        err = np.max(np.abs(y_sor - y_exact))
        print(f"{n:7d}  {h:10.4e}  {h/eps:10.4e}  {err:12.4e}  {it_sor:10d}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Part 1: fixed n=100, varying eps")
    print("=" * 60)
    for eps in [1, 0.1, 0.01, 0.0001]:
        solve(eps)

    print("=" * 60)
    print("Part 2: for each eps, vary n to observe discretization error")
    print("=" * 60)
    for eps in [1, 0.1, 0.01, 0.0001]:
        solve_vary_n(eps)
