import numpy as np

from src.linalg_utils import (
    dominant_eigenpair,
    matrix_condition_number,
    relative_residual,
    solve_linear_system,
)


def main() -> None:
    a = np.array(
        [
            [4.0, 1.0, -1.0],
            [2.0, 7.0, 1.0],
            [1.0, -3.0, 12.0],
        ]
    )
    b = np.array([3.0, 19.0, 31.0])

    x = solve_linear_system(a, b)
    res = relative_residual(a, x, b)
    cond = matrix_condition_number(a)
    eigval, eigvec = dominant_eigenpair(a)

    print("Solution x:", x)
    print("Relative residual:", res)
    print("Condition number kappa_2(A):", cond)
    print("Dominant eigenvalue:", eigval)
    print("Associated eigenvector:", eigvec)


if __name__ == "__main__":
    main()
