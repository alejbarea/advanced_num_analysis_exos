import numpy as np

from src.linalg_utils import (
    matrix_condition_number,
    relative_residual,
    solve_linear_system,
)


def test_solve_linear_system_exact_diagonal() -> None:
    a = np.diag([2.0, 3.0, 4.0])
    b = np.array([2.0, 6.0, 8.0])
    x = solve_linear_system(a, b)
    assert np.allclose(x, np.array([1.0, 2.0, 2.0]))


def test_relative_residual_is_small() -> None:
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    x = solve_linear_system(a, b)
    assert relative_residual(a, x, b) < 1e-12


def test_condition_number_identity() -> None:
    a = np.eye(4)
    assert np.isclose(matrix_condition_number(a), 1.0)
