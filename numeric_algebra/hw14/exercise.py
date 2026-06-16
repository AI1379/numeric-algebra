import numpy as np


def _householder_vector(x: np.ndarray):
    """Return v, beta so that (I - beta v v*) x is a multiple of e1."""
    x = np.asarray(x, dtype=complex).copy()
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return x, 0.0

    phase = x[0] / abs(x[0]) if abs(x[0]) > 0 else 1.0 + 0.0j
    alpha = -phase * norm_x
    v = x.copy()
    v[0] -= alpha
    beta = 2.0 / np.vdot(v, v)
    return v, beta


def hessenberg_reduction(A: np.ndarray):
    """Householder reduction A -> Q* H Q, with H upper Hessenberg."""
    H = np.asarray(A, dtype=complex).copy()
    n = H.shape[0]
    Q = np.eye(n, dtype=complex)

    for k in range(n - 2):
        x = H[k + 1 :, k]
        v, beta = _householder_vector(x)
        if beta == 0:
            continue

        H[k + 1 :, k:] -= beta * np.outer(v, np.conj(v) @ H[k + 1 :, k:])
        H[:, k + 1 :] -= beta * np.outer(H[:, k + 1 :] @ v, np.conj(v))
        Q[:, k + 1 :] -= beta * np.outer(Q[:, k + 1 :] @ v, np.conj(v))

    _clean_hessenberg(H)
    return H, Q


def _clean_hessenberg(H: np.ndarray, tol: float = 1e-14):
    n = H.shape[0]
    for i in range(n):
        for j in range(i - 1):
            if abs(H[i, j]) < tol:
                H[i, j] = 0.0


def _wilkinson_shift(H: np.ndarray, lo: int, hi: int):
    if hi == lo:
        return H[hi, hi]

    a = H[hi - 1, hi - 1]
    b = H[hi - 1, hi]
    c = H[hi, hi - 1]
    d = H[hi, hi]
    half_trace = 0.5 * (a + d)
    delta = np.sqrt(0.25 * (a - d) ** 2 + b * c)
    mu1 = half_trace + delta
    mu2 = half_trace - delta
    return mu1 if abs(mu1 - d) <= abs(mu2 - d) else mu2


def _implicit_single_shift_qr_step(
    H: np.ndarray, Z: np.ndarray, lo: int, hi: int, shift: complex
):
    """One implicit shifted QR step on H[lo:hi+1, lo:hi+1]."""
    for k in range(lo, hi):
        if k == lo:
            x = np.array([H[lo, lo] - shift, H[lo + 1, lo]], dtype=complex)
        else:
            x = np.array([H[k, k - 1], H[k + 1, k - 1]], dtype=complex)

        v, beta = _householder_vector(x)
        if beta == 0:
            continue

        rows = slice(k, k + 2)
        H[rows, :] -= beta * np.outer(v, np.conj(v) @ H[rows, :])
        H[:, rows] -= beta * np.outer(H[:, rows] @ v, np.conj(v))
        Z[:, rows] -= beta * np.outer(Z[:, rows] @ v, np.conj(v))

        if k > lo:
            H[k + 1, k - 1] = 0.0

    _clean_hessenberg(H)


def _deflate(H: np.ndarray, tol: float):
    n = H.shape[0]
    for i in range(1, n):
        threshold = tol * (abs(H[i - 1, i - 1]) + abs(H[i, i]) + 1.0)
        if abs(H[i, i - 1]) <= threshold:
            H[i, i - 1] = 0.0


def _schur_eigenvectors(T: np.ndarray, Z: np.ndarray, eigvals: np.ndarray):
    n = T.shape[0]
    vectors = np.zeros((n, n), dtype=complex)
    eps = 1e-12 * max(1.0, np.linalg.norm(T, ord=np.inf))

    for j, lam in enumerate(eigvals):
        y = np.zeros(n, dtype=complex)
        y[j] = 1.0
        for i in range(j - 1, -1, -1):
            rhs = T[i, i + 1 : j + 1] @ y[i + 1 : j + 1]
            denom = T[i, i] - lam
            if abs(denom) < eps:
                denom = eps
            y[i] = -rhs / denom

        v = Z @ y
        norm_v = np.linalg.norm(v)
        vectors[:, j] = v / norm_v if norm_v != 0 else v

    return vectors


def implicit_qr_eig(
    A: np.ndarray,
    tol: float = 1e-12,
    max_iter: int | None = None,
    compute_vectors: bool = True,
):
    """Compute all eigenvalues and right eigenvectors by implicit shifted QR.

    The implementation first reduces the real matrix to upper Hessenberg form,
    then applies complex single-shift Francis steps with Wilkinson shifts.
    """
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]
    if max_iter is None:
        max_iter = 5000 * max(1, n)

    H, Z = hessenberg_reduction(A)
    iterations = 0
    hi = n - 1

    while hi > 0:
        _deflate(H, tol)
        while hi > 0 and H[hi, hi - 1] == 0:
            hi -= 1
        if hi == 0:
            break

        lo = hi - 1
        while lo > 0 and H[lo, lo - 1] != 0:
            lo -= 1

        shift = _wilkinson_shift(H, lo, hi)
        _implicit_single_shift_qr_step(H, Z, lo, hi, shift)

        iterations += 1
        if iterations > max_iter:
            raise RuntimeError("implicit QR iteration did not converge")

    T = np.triu(H)
    eigvals = np.diag(T).copy()
    eigvecs = _schur_eigenvectors(T, Z, eigvals) if compute_vectors else None
    return eigvals, eigvecs, T, iterations


def poly_to_companion(coeffs: np.ndarray):
    """Companion matrix for x^n + c[n-1]x^(n-1) + ... + c[0]."""
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs)
    C = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        C[i, i - 1] = 1.0
    C[:, -1] = -coeffs
    return C


def _sort_complex(values: np.ndarray):
    return sorted(
        values, key=lambda z: (round(float(z.real), 12), round(float(z.imag), 12))
    )


def _fmt_complex(z: complex, digits: int = 12):
    if abs(z.imag) < 5e-11:
        return f"{z.real:+.{digits}f}"
    if abs(z.real) < 5e-11:
        return f"{z.imag:+.{digits}f}i"
    return f"{z.real:+.{digits}f}{z.imag:+.{digits}f}i"


def _print_eigenpairs(A: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray):
    order = np.argsort(eigvals.real + 1e-7 * eigvals.imag)
    for idx in order:
        lam = eigvals[idx]
        v = eigvecs[:, idx]
        residual = np.linalg.norm(A @ v - lam * v)
        print(f"  lambda = {_fmt_complex(lam)}   residual = {residual:.3e}")
        print(f"    v = {np.array2string(v, precision=6, suppress_small=True)}")


def exercise2_roots():
    print("=" * 70)
    print("Exercise 2: roots of x^41 + x^3 + 1 = 0")
    print("=" * 70)
    coeffs = np.zeros(41)
    coeffs[0] = 1.0
    coeffs[3] = 1.0
    C = poly_to_companion(coeffs)
    eigvals, _, _, iterations = implicit_qr_eig(C, tol=1e-11, compute_vectors=False)

    print(f"implicit QR iterations: {iterations}")
    print("roots:")
    for root in _sort_complex(eigvals):
        value = root**41 + root**3 + 1.0
        print(f"  {_fmt_complex(root)}   |p(root)| = {abs(value):.3e}")
    print()


def exercise3_parameter_matrix():
    print("=" * 70)
    print("Exercise 3: eigenvalues of A for x = 0.9, 1.0, 1.1")
    print("=" * 70)

    for x in [0.9, 1.0, 1.1]:
        A = np.array(
            [
                [9.1, 3.0, 2.6, 4.0],
                [4.2, 5.3, 1.7, 1.6],
                [3.2, 1.7, 9.4, x],
                [6.1, 4.9, 3.5, 6.2],
            ],
            dtype=float,
        )
        eigvals, eigvecs, _, iterations = implicit_qr_eig(A, tol=1e-12)

        print(f"\nx = {x:.1f}, implicit QR iterations: {iterations}")
        _print_eigenpairs(A.astype(complex), eigvals, eigvecs)


if __name__ == "__main__":
    exercise2_roots()
    exercise3_parameter_matrix()
