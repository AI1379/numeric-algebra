import numpy as np
import sympy as sp
from typing import Union


def mat_to_latex(mat: Union[np.ndarray, sp.Matrix], precision=2, environment="bmatrix"):
    """
    Convert a matrix to a LaTeX string.

    Parameters:
    mat (Union[np.ndarray, sp.Matrix]): The input matrix.
    precision (int): The number of decimal places to round to.
    environment (str): The LaTeX environment to use (e.g., "bmatrix", "pmatrix").

    Returns:
    str: A LaTeX string representing the matrix.
    """
    if isinstance(mat, sp.Matrix):
        return sp.latex(mat)
    elif isinstance(mat, np.ndarray):
        rounded_mat = np.round(mat, precision)
        latex_str = f"\\begin{{{environment}}}\n"
        for row in rounded_mat:
            latex_str += " & ".join(map(str, row)) + " \\\\\n"
        latex_str += f"\\end{{{environment}}}"
        return latex_str
    else:
        raise TypeError("Input must be a numpy array or a sympy Matrix.")
