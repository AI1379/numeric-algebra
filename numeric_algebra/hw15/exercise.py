import numpy as np


def off_diagonal_norm(A: np.ndarray):
    off = A - np.diag(np.diag(A))
    return np.linalg.norm(off, ord="fro")


def _jacobi_rotate(A: np.ndarray, V: np.ndarray, p: int, q: int):
    app = A[p, p]
    aqq = A[q, q]
    apq = A[p, q]
    if apq == 0.0:
        return

    tau = (aqq - app) / (2.0 * apq)
    if tau >= 0.0:
        t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
    else:
        t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
    c = 1.0 / np.sqrt(1.0 + t * t)
    s = t * c

    n = A.shape[0]
    for r in range(n):
        if r == p or r == q:
            continue
        arp = A[r, p]
        arq = A[r, q]
        A[r, p] = c * arp - s * arq
        A[p, r] = A[r, p]
        A[r, q] = s * arp + c * arq
        A[q, r] = A[r, q]

    A[p, p] = app - t * apq
    A[q, q] = aqq + t * apq
    A[p, q] = 0.0
    A[q, p] = 0.0

    vp = V[:, p].copy()
    vq = V[:, q].copy()
    V[:, p] = c * vp - s * vq
    V[:, q] = s * vp + c * vq


def threshold_jacobi_eig(
    A: np.ndarray,
    tol: float = 1e-12,
    max_sweeps: int = 200,
):
    """Compute all eigenpairs of a real symmetric matrix by threshold Jacobi.

    The routine repeatedly scans the upper triangle and applies a Jacobi
    rotation only when |a[p,q]| is above the current pass threshold.
    It does not call np.linalg.eig or np.linalg.eigh.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("A must be real symmetric")

    B = A.copy()
    n = B.shape[0]
    V = np.eye(n)
    scale = max(1.0, np.linalg.norm(B, ord="fro"))
    stop = tol * scale

    sweeps = 0
    rotations = 0
    while off_diagonal_norm(B) > stop:
        if sweeps >= max_sweeps:
            raise RuntimeError("threshold Jacobi iteration did not converge")

        threshold = off_diagonal_norm(B) / (4.0 * n)
        sweep_rotations = 0
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(B[p, q]) >= threshold:
                    _jacobi_rotate(B, V, p, q)
                    rotations += 1
                    sweep_rotations += 1

        sweeps += 1
        if sweep_rotations == 0:
            break

    eigvals = np.diag(B).copy()
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = V[:, order]

    for j in range(n):
        pivot = np.argmax(np.abs(eigvecs[:, j]))
        if eigvecs[pivot, j] < 0.0:
            eigvecs[:, j] *= -1.0

    return eigvals, eigvecs, sweeps, rotations, off_diagonal_norm(B)


def tridiagonal_toeplitz(n: int, diag: float, offdiag: float):
    A = np.diag(np.full(n, diag))
    A += np.diag(np.full(n - 1, offdiag), 1)
    A += np.diag(np.full(n - 1, offdiag), -1)
    return A


def exact_toeplitz_eigenpairs(n: int, diag: float, offdiag: float):
    k = np.arange(n, 0, -1)
    eigvals = diag + 2.0 * offdiag * np.cos(k * np.pi / (n + 1))
    rows = np.arange(1, n + 1)[:, None]
    eigvecs = np.sqrt(2.0 / (n + 1)) * np.sin(rows * k[None, :] * np.pi / (n + 1))

    for j in range(n):
        pivot = np.argmax(np.abs(eigvecs[:, j]))
        if eigvecs[pivot, j] < 0.0:
            eigvecs[:, j] *= -1.0

    return eigvals, eigvecs


def residual_report(A: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray):
    residuals = np.linalg.norm(A @ eigvecs - eigvecs * eigvals, axis=0)
    orth_error = np.linalg.norm(eigvecs.T @ eigvecs - np.eye(A.shape[0]), ord=np.inf)
    return np.max(residuals), orth_error


def eigenvector_error(eigvecs: np.ndarray, exact_vecs: np.ndarray):
    errors = []
    for j in range(eigvecs.shape[1]):
        errors.append(
            min(
                np.linalg.norm(eigvecs[:, j] - exact_vecs[:, j]),
                np.linalg.norm(eigvecs[:, j] + exact_vecs[:, j]),
            )
        )
    return max(errors)


def print_edge_eigenvalues(eigvals: np.ndarray, count: int = 4):
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
        print(
            f"    first {components} entries: "
            f"{np.array2string(head, precision=6, suppress_small=True)}"
        )
        if n > components:
            print(
                f"    last  {components} entries: "
                f"{np.array2string(tail, precision=6, suppress_small=True)}"
            )


def run_case(n: int, show_vectors: bool = False):
    A = tridiagonal_toeplitz(n, 4.0, 1.0)
    eigvals, eigvecs, sweeps, rotations, final_off = threshold_jacobi_eig(A)
    exact_vals, exact_vecs = exact_toeplitz_eigenpairs(n, 4.0, 1.0)

    eig_error = np.max(np.abs(eigvals - exact_vals))
    vec_error = eigenvector_error(eigvecs, exact_vecs)
    max_residual, orth_error = residual_report(A, eigvals, eigvecs)

    print(f"A = tridiag(1, 4, 1), n = {n}")
    print(f"  Jacobi sweeps       : {sweeps}")
    print(f"  Jacobi rotations    : {rotations}")
    print(f"  final offdiag norm  : {final_off:.3e}")
    print(f"  max eigenvalue error: {eig_error:.3e}")
    print(f"  max eigenvector err : {vec_error:.3e}")
    print(f"  max residual        : {max_residual:.3e}")
    print(f"  orthogonality error : {orth_error:.3e}")
    print_edge_eigenvalues(eigvals)

    if show_vectors:
        print_selected_eigenvectors(eigvals, eigvecs, [0, n // 2, n - 1])
    print()

    return eigvals, eigvecs


def main():
    print("=" * 72)
    print("Threshold Jacobi method for real symmetric tridiagonal matrices")
    print("A = tridiag(1, 4, 1), n = 50, 51, ..., 100")
    print("=" * 72)
    print(
        "The program computes all eigenvalues and eigenvectors for every n. "
        "For compact output, only edge eigenvalues and selected eigenvectors "
        "are printed."
    )
    print()

    for n in range(50, 101):
        run_case(n, show_vectors=(n in [50, 75, 100]))


if __name__ == "__main__":
    main()
