import numpy as np


def _householder_vector(x: np.ndarray):
    """Return v, beta, alpha with (I - beta vv^T)x = alpha e1."""
    x = np.asarray(x, dtype=float).copy()
    norm_x = np.linalg.norm(x)
    if norm_x == 0.0:
        return x, 0.0, 0.0

    sign = 1.0 if x[0] >= 0.0 else -1.0
    alpha = -sign * norm_x
    v = x.copy()
    v[0] -= alpha
    beta = 2.0 / (v @ v)
    return v, beta, alpha


def _clean_tridiagonal(T: np.ndarray, tol: float = 1e-14):
    n = T.shape[0]
    for i in range(n):
        for j in range(i - 1):
            if abs(T[i, j]) <= tol:
                T[i, j] = 0.0
                T[j, i] = 0.0
    T[:] = 0.5 * (T + T.T)


def symmetric_tridiagonal_reduction(A: np.ndarray):
    """Householder reduction A = Q T Q^T, where T is real tridiagonal."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("A must be real symmetric")

    n = A.shape[0]
    T = A.copy()
    Q = np.eye(n)

    for k in range(n - 2):
        v, beta, alpha = _householder_vector(T[k + 1 :, k])
        if beta == 0.0:
            continue

        sub = T[k + 1 :, k + 1 :]
        w = beta * (sub @ v)
        tau = -0.5 * beta * (v @ w)
        w += tau * v
        sub -= np.outer(v, w) + np.outer(w, v)

        T[k + 1, k] = alpha
        T[k, k + 1] = alpha
        T[k + 2 :, k] = 0.0
        T[k, k + 2 :] = 0.0

        Q[:, k + 1 :] -= beta * np.outer(Q[:, k + 1 :] @ v, v)

    _clean_tridiagonal(T)
    return T, Q


def _givens(a: float, b: float):
    if b == 0.0:
        return 1.0, 0.0
    r = np.hypot(a, b)
    return a / r, b / r


def _wilkinson_shift(T: np.ndarray, hi: int):
    a = T[hi - 1, hi - 1]
    b = T[hi, hi - 1]
    d = T[hi, hi]
    delta = 0.5 * (a - d)
    sign = 1.0 if delta >= 0.0 else -1.0
    return d - sign * b**2 / (abs(delta) + np.sqrt(delta**2 + b**2))


def _apply_givens_similarity(T: np.ndarray, Q: np.ndarray, k: int, c: float, s: float):
    i, j = k, k + 1

    row_i = T[i, :].copy()
    row_j = T[j, :].copy()
    T[i, :] = c * row_i + s * row_j
    T[j, :] = -s * row_i + c * row_j

    col_i = T[:, i].copy()
    col_j = T[:, j].copy()
    T[:, i] = c * col_i + s * col_j
    T[:, j] = -s * col_i + c * col_j

    qi = Q[:, i].copy()
    qj = Q[:, j].copy()
    Q[:, i] = c * qi + s * qj
    Q[:, j] = -s * qi + c * qj


def _implicit_symmetric_qr_step(T: np.ndarray, Q: np.ndarray, lo: int, hi: int, shift: float):
    """One implicit shifted QR step on the unreduced tridiagonal block lo:hi."""
    x = T[lo, lo] - shift
    z = T[lo + 1, lo]

    for k in range(lo, hi):
        c, s = _givens(x, z)
        _apply_givens_similarity(T, Q, k, c, s)

        if k > lo:
            T[k + 1, k - 1] = 0.0
            T[k - 1, k + 1] = 0.0

        if k < hi - 1:
            x = T[k + 1, k]
            z = T[k + 2, k]

    _clean_tridiagonal(T)


def _is_deflatable(T: np.ndarray, i: int, tol: float):
    threshold = tol * (abs(T[i - 1, i - 1]) + abs(T[i, i]) + 1.0)
    return abs(T[i, i - 1]) <= threshold


def implicit_symmetric_qr(
    A: np.ndarray,
    tol: float = 1e-12,
    max_iter: int | None = None,
):
    """Compute all eigenvalues and eigenvectors of a real symmetric matrix.

    The routine uses Householder tridiagonalization and implicit shifted QR
    steps with Wilkinson shifts. It does not call np.linalg.eig/eigh.
    """
    T, Q = symmetric_tridiagonal_reduction(A)
    n = T.shape[0]
    if max_iter is None:
        max_iter = 5000 * max(1, n)

    iterations = 0
    hi = n - 1
    while hi > 0:
        for i in range(1, hi + 1):
            if _is_deflatable(T, i, tol):
                T[i, i - 1] = 0.0
                T[i - 1, i] = 0.0

        while hi > 0 and T[hi, hi - 1] == 0.0:
            hi -= 1
        if hi == 0:
            break

        lo = hi - 1
        while lo > 0 and T[lo, lo - 1] != 0.0:
            lo -= 1

        shift = _wilkinson_shift(T, hi)
        _implicit_symmetric_qr_step(T, Q, lo, hi, shift)

        iterations += 1
        if iterations > max_iter:
            raise RuntimeError("implicit symmetric QR iteration did not converge")

    eigvals = np.diag(T).copy()
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = Q[:, order]

    for j in range(n):
        pivot = np.argmax(np.abs(eigvecs[:, j]))
        if eigvecs[pivot, j] < 0.0:
            eigvecs[:, j] *= -1.0

    return eigvals, eigvecs, iterations


def tridiagonal_toeplitz(n: int, diag: float, offdiag: float):
    A = np.diag(np.full(n, diag))
    A += np.diag(np.full(n - 1, offdiag), 1)
    A += np.diag(np.full(n - 1, offdiag), -1)
    return A


def exact_toeplitz_eigenvalues(n: int, diag: float, offdiag: float):
    k = np.arange(1, n + 1)
    return np.sort(diag + 2.0 * offdiag * np.cos(k * np.pi / (n + 1)))


def residual_report(A: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray):
    residuals = np.linalg.norm(A @ eigvecs - eigvecs * eigvals, axis=0)
    orth_error = np.linalg.norm(eigvecs.T @ eigvecs - np.eye(A.shape[0]), ord=np.inf)
    return np.max(residuals), orth_error


def print_edge_eigenvalues(eigvals: np.ndarray, count: int = 5):
    n = len(eigvals)
    if n <= 2 * count:
        print("  eigenvalues:", np.array2string(eigvals, precision=12))
        return

    print("  smallest eigenvalues:", np.array2string(eigvals[:count], precision=12))
    print("  largest  eigenvalues:", np.array2string(eigvals[-count:], precision=12))


def print_selected_eigenvectors(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    indices: list[int],
    components: int = 8,
):
    n = eigvecs.shape[0]
    for idx in indices:
        head = eigvecs[:components, idx]
        tail = eigvecs[-components:, idx]
        print(f"  eigenvector for lambda[{idx}] = {eigvals[idx]:.12f}")
        print(f"    first {components} entries: {np.array2string(head, precision=6, suppress_small=True)}")
        if n > components:
            print(f"    last  {components} entries: {np.array2string(tail, precision=6, suppress_small=True)}")


def run_case(name: str, n: int, diag: float, offdiag: float, show_vectors: bool = False):
    A = tridiagonal_toeplitz(n, diag, offdiag)
    eigvals, eigvecs, iterations = implicit_symmetric_qr(A)
    exact = exact_toeplitz_eigenvalues(n, diag, offdiag)
    max_residual, orth_error = residual_report(A, eigvals, eigvecs)
    eig_error = np.max(np.abs(eigvals - exact))

    print(f"{name}, n = {n}")
    print(f"  QR iterations       : {iterations}")
    print(f"  max eigenvalue error: {eig_error:.3e}")
    print(f"  max residual        : {max_residual:.3e}")
    print(f"  orthogonality error : {orth_error:.3e}")
    print_edge_eigenvalues(eigvals)

    if show_vectors:
        print_selected_eigenvectors(eigvals, eigvecs, [0, n // 2, n - 1])
    print()

    return eigvals, eigvecs


def exercise_1_2():
    print("=" * 72)
    print("Textbook p.244, Exercise 1(2): diag = 4, offdiag = 1")
    print("=" * 72)
    for n in range(50, 101):
        run_case("A = tridiag(1, 4, 1)", n, 4.0, 1.0, show_vectors=(n == 100))


def exercise_2_2():
    print("=" * 72)
    print("Textbook p.244, Exercise 2(2): diag = 2, offdiag = -1, n = 100")
    print("=" * 72)
    run_case("A = tridiag(-1, 2, -1)", 100, 2.0, -1.0, show_vectors=True)


if __name__ == "__main__":
    exercise_1_2()
    exercise_2_2()
