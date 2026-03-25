import numpy as np
import sympy as sp
from numeric_algebra.hw3.cholesky import cholesky_sp, cholesky_np_im, cholesky_sp_im
from numeric_algebra.common.gaussian_jordan_inv import (
    gaussian_jordan_inv_no_pivot,
    gaussian_jordan_inv_col_pivot,
)


def question1():
    A = sp.Matrix(
        [
            [4, -2, 4, 2],
            [-2, 10, -2, -7],
            [4, -2, 8, 4],
            [2, -7, 4, 7],
        ]
    )
    b = sp.Matrix([8, 2, 16, 6])
    L = cholesky_sp(A)
    sp.pprint(L)
    sp.pprint(L * L.T - A)

    y = L.inv() * b
    sp.pprint(y)

    x = L.T.inv() * y
    sp.pprint(x)

    print("\nLaTeX representation of L:")
    print(sp.latex(L))
    print("\nLaTeX representation of the solution y:")
    print(sp.latex(y))
    print("\nLaTeX representation of the solution x:")
    print(sp.latex(x))


def question2():
    A = sp.Matrix(
        [
            [10, 20, 30],
            [20, 45, 80],
            [30, 80, 171],
        ]
    )
    b = sp.Matrix([10, 5, -31])
    L, D = cholesky_sp_im(A)
    sp.pprint(L)
    sp.pprint(D)
    sp.pprint(L * D * L.T - A)
    y = L.inv() * b
    sp.pprint(y)
    DLT = D * L.T
    x = DLT.inv() * y
    sp.pprint(x)

    # Check the solution
    Ax = A * x
    sp.pprint(Ax - b)

    print("\nLaTeX representation of L:")
    print(sp.latex(L))
    print("\nLaTeX representation of D:")
    print(sp.latex(D))
    print("\nLaTeX representation of the solution y:")
    print(sp.latex(y))
    print("\nLaTeX representation of the solution x:")
    print(sp.latex(x))


def question4_symbolic():
    """
    用 SymPy 实现无主元 Gauss-Jordan 消元法，详细输出每一步的过程
    """
    print("\n" + "=" * 80)
    print("题目 4：Gauss-Jordan 法求矩阵逆（详细过程）")
    print("=" * 80)
    
    A = sp.Matrix(
        [
            [2, 1, -3, -1],
            [3, 1, 0, 7],
            [-1, 2, 4, -2],
            [1, 0, -1, 5],
        ]
    )
    
    n = A.shape[0]
    
    # 构造增广矩阵 [A | I]
    aug = A.row_join(sp.eye(n))
    
    print("\n初始增广矩阵 [A | I]:")
    sp.pprint(aug)
    
    # Gauss-Jordan 消元
    for k in range(n):
        print(f"\n【第 {k+1} 步：处理第 {k+1} 列】")
        
        # 主元行归一化
        if aug[k, k] == 0:
            print(f"警告：第 {k} 行第 {k} 列的主元为 0")
            continue
        
        pivot = aug[k, k]
        print(f"主元为 {pivot}，将第 {k+1} 行除以 {pivot}：")
        aug[k, :] = aug[k, :] / pivot
        sp.pprint(aug)
        
        # 消去其他行
        print(f"消去其他行的第 {k+1} 列元素：")
        for i in range(n):
            if i != k:
                factor = aug[i, k]
                if factor != 0:
                    print(f"  第 {i+1} 行 := 第 {i+1} 行 - ({factor}) * 第 {k+1} 行")
                    aug[i, :] = aug[i, :] - factor * aug[k, :]
        
        sp.pprint(aug)
    
    # 提取结果
    A_inv = aug[:, n:]
    
    print("\n最终化简行阶梯形 [I | A^{-1}]:")
    sp.pprint(aug)
    
    print("\n逆矩阵 A^{-1}:")
    sp.pprint(A_inv)
    
    # 验证
    print("\n验证 A @ A^{-1} = I:")
    product = A * A_inv
    sp.pprint(product)
    residual = sp.simplify(product - sp.eye(n))
    sp.pprint(residual)
    
    print("\nLaTeX 代码:")
    print("A^{-1} = ")
    print(sp.latex(A_inv))
    
    return A_inv


def question4():
    """
    测试 Gauss-Jordan 消元法（无主元和列主元）求矩阵逆
    """
    A = np.array(
        [
            [2, 1, -3, -1],
            [3, 1, 0, 7],
            [-1, 2, 4, -2],
            [1, 0, -1, 5],
        ],
        dtype=float,
    )

    print("=" * 60)
    print("题目 4：Gauss-Jordan 法求矩阵逆")
    print("=" * 60)

    print("\n原矩阵 A:")
    print(A)

    # 无主元版本
    print("\n【无主元 Gauss-Jordan 消元法】")
    try:
        A_inv_no_pivot = gaussian_jordan_inv_no_pivot(A)
        print("A^{-1} (无主元):")
        print(A_inv_no_pivot)

        # 验证
        product = A @ A_inv_no_pivot
        print("\nA @ A^{-1} (应为单位矩阵):")
        print(product)
        print(f"残差范数: {np.linalg.norm(product - np.eye(4)):.2e}")
    except Exception as e:
        print(f"错误: {e}")
        return

    # 列主元版本
    print("\n【列主元 Gauss-Jordan 消元法】")
    try:
        A_inv_col_pivot = gaussian_jordan_inv_col_pivot(A)
        print("A^{-1} (列主元):")
        print(A_inv_col_pivot)

        # 验证
        product = A @ A_inv_col_pivot
        print("\nA @ A^{-1} (应为单位矩阵):")
        print(product)
        print(f"残差范数: {np.linalg.norm(product - np.eye(4)):.2e}")
    except Exception as e:
        print(f"错误: {e}")
        return

    # 两个版本的对比
    print("\n【两个版本的对比】")
    if "A_inv_no_pivot" in locals() and "A_inv_col_pivot" in locals():
        diff = np.linalg.norm(A_inv_no_pivot - A_inv_col_pivot)
        print(f"两个版本的逆矩阵差异: {diff:.2e}")


if __name__ == "__main__":
    question1()
    question2()
    question4_symbolic()
    question4()
