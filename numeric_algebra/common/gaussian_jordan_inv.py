import numpy as np


def gaussian_jordan_inv_no_pivot(A: np.ndarray) -> np.ndarray:
    """
    使用 Gauss-Jordan 消元法（无主元）求矩阵的逆。

    Args:
        A: 非奇异 n x n 矩阵

    Returns:
        A 的逆矩阵 A^{-1}
    """
    A = A.copy().astype(float)
    n = A.shape[0]

    # 构造增广矩阵 [A | I]
    aug = np.hstack([A, np.eye(n)])

    # 前向消元 + 回代，同时化为化简行阶梯形
    for k in range(n):
        # 选择主元（这里无主元版本直接取对角线元素）
        if np.abs(aug[k, k]) < 1e-14:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        # 主元行归一化
        aug[k, :] /= aug[k, k]

        # 消去上下的其他行
        for i in range(n):
            if i != k:
                aug[i, :] -= aug[i, k] * aug[k, :]

    # 提取右半部分作为逆矩阵
    A_inv = aug[:, n:]
    return A_inv


def gaussian_jordan_inv_col_pivot(A: np.ndarray) -> np.ndarray:
    """
    使用 Gauss-Jordan 消元法（列主元）求矩阵的逆。

    Args:
        A: 非奇异 n x n 矩阵

    Returns:
        A 的逆矩阵 A^{-1}
    """
    A = A.copy().astype(float)
    n = A.shape[0]

    # 构造增广矩阵 [A | I]
    aug = np.hstack([A, np.eye(n)])

    # 记录行交换用于后续恢复
    row_perm = np.arange(n)

    # 前向消元 + 回代，同时化为化简行阶梯形
    for k in range(n):
        # 列主元选择：从第 k 行到第 n-1 行中找最大绝对值
        idx_max = k + np.argmax(np.abs(aug[k:n, k]))

        if np.abs(aug[idx_max, k]) < 1e-14:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        # 行交换
        if idx_max != k:
            aug[[k, idx_max], :] = aug[[idx_max, k], :]
            row_perm[[k, idx_max]] = row_perm[[idx_max, k]]

        # 主元行归一化
        aug[k, :] /= aug[k, k]

        # 消去上下的其他行
        for i in range(n):
            if i != k:
                aug[i, :] -= aug[i, k] * aug[k, :]

    # 提取右半部分作为逆矩阵
    A_inv = aug[:, n:]
    return A_inv
