import numpy as np


def power_method(
    A: np.ndarray, v0: np.ndarray = None, max_iter: int = 1000, eps: float = 1e-10
):
    n = A.shape[0]
    if v0 is None:
        v0 = np.ones(n)
    v = v0 / np.linalg.norm(v0, np.inf)

    mu_prev = None
    for k in range(max_iter):
        y = A @ v
        mu = y[np.argmax(np.abs(y))]
        v = y / mu
        if mu_prev is not None and abs(abs(mu) - abs(mu_prev)) < eps:
            break
        mu_prev = mu

    # Rayleigh quotient for sign
    eigenvalue = float(v @ (A @ v) / (v @ v))
    return eigenvalue, v, k + 1


def poly_to_companion(coeffs: np.ndarray) -> np.ndarray:
    # coeffs = [a_0, a_1, ..., a_{n-1}] for x^n + a_{n-1}x^{n-1} + ... + a_0
    n = len(coeffs)
    C = np.zeros((n, n))
    for i in range(1, n):
        C[i, i - 1] = 1.0
    C[:, -1] = -coeffs
    return C


def solve_poly(coeffs: np.ndarray, label: str):
    print("=" * 60)
    print(label)
    C = poly_to_companion(coeffs)
    eigenvalue, v, iters = power_method(C)
    print(f"  Dominant root:   {eigenvalue:+.10f}")
    print(f"  Eigenvector:     {v}")
    print(f"  Iterations:      {iters}")
    eigvals = np.linalg.eigvals(C)
    print("  NumPy eigenvalues (sorted by |eig|):")
    for ev in sorted(eigvals, key=lambda x: -abs(x)):
        if abs(ev.imag) < 1e-10:
            print(f"    {ev.real:+.10f}  (|eig| = {abs(ev):.10f})")
        else:
            print(f"    {ev.real:+.10f} {ev.imag:+.10f}i  (|eig| = {abs(ev):.10f})")
    print()


if __name__ == "__main__":
    solve_poly(np.array([3.0, -5.0, 1.0]), "(i) x^3 + x^2 - 5x + 3 = 0")

    solve_poly(np.array([-1.0, -3.0, 0.0]), "(ii) x^3 - 3x - 1 = 0")

    solve_poly(
        np.array([-1000.0, 790.0, -99902.0, 79108.9, 9802.08, 10891.01, 208.01, 101.0]),
        "(iii) x^8 + 101x^7 + 208.01x^6 + 10891.01x^5"
        "\n      + 9802.08x^4 + 79108.9x^3 - 99902x^2 + 790x - 1000 = 0",
    )
