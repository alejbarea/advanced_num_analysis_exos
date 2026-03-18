import numpy as np


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using a direct solver."""
    return np.linalg.solve(a, b)


def relative_residual(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Compute ||Ax - b|| / ||b|| in 2-norm."""
    numerator = np.linalg.norm(a @ x - b, ord=2)
    denominator = np.linalg.norm(b, ord=2)
    if denominator == 0.0:
        return float(numerator)
    return float(numerator / denominator)


def matrix_condition_number(a: np.ndarray) -> float:
    """Compute the condition number kappa_2(A)."""
    return float(np.linalg.cond(a, p=2))


def dominant_eigenpair(a: np.ndarray) -> tuple[complex, np.ndarray]:
    """Return eigenvalue with largest magnitude and its eigenvector."""
    eigvals, eigvecs = np.linalg.eig(a)
    idx = int(np.argmax(np.abs(eigvals)))
    return eigvals[idx], eigvecs[:, idx]
