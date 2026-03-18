import numpy as np

def is_upper_hessenberg(matrix: np.ndarray) -> bool:
    """
    Check if the given matrix is an upper Hessenberg matrix.

    An upper Hessenberg matrix is a square matrix where all entries below the first subdiagonal are zero.
    In other words, for a matrix A, A[i, j] = 0 for all i > j + 1.

    Parameters:
    matrix (np.ndarray): The input matrix to check.

    Returns:
    bool: True if the matrix is an upper Hessenberg matrix, False otherwise.
    """
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if i > j + 1 and matrix[i, j] != 0:
                return False
    return True

def get_givens_rotation(a, b):
    r = np.hypot(a, b)
    if r == 0:
        return 1.0, 0.0
    return a / r, -b / r

def hessqr(matrix: np.ndarray):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    elif matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")
    elif not is_upper_hessenberg(matrix):
        raise ValueError("Input must be an upper Hessenberg matrix.")
    n = matrix.shape[0]
    R = matrix.astype(float).copy()
    Q_T = np.eye(n) # Initialize Q as the identity matrix
    for k in range(n - 1):
        a = R[k, k]
        b = R[k + 1, k]
        c, s = get_givens_rotation(a, b)
        G_2x2 = np.array([[c, -s], 
                          [s,  c]])
        R[k:k+2, k:] = G_2x2 @ R[k:k+2, k:]
        Q_T[k:k+2, :] = G_2x2 @ Q_T[k:k+2, :]
    return Q_T.T, R


def define_householder_reflector_real(matrix: np.ndarray, k: int):
    n = matrix.shape[0]
    x = matrix[k + 1:, k]
    if np.linalg.norm(x[1:]) < 1e-12:
        return np.eye(n)
    sign_x = np.sign(x[0])
    if sign_x == 0:
        sign_x = 1.0
    alpha = -sign_x * np.linalg.norm(x)
    r = np.sqrt(0.5 * (alpha**2 - matrix[k + 1, k] * alpha))
    v = np.zeros(matrix.shape[0])
    v[k + 1] = (matrix[k + 1, k] - alpha) / (2 * r)
    v[k + 2:] = matrix[k + 2:, k] / (2 * r)
    return v


def matrix_to_hess_real(matrix: np.ndarray):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    elif matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")
    n = matrix.shape[0]
    H = matrix.astype(float).copy()
    Q = np.eye(n) # Initialize Q as the identity matrix
    for k in range(n - 2):
        v = define_householder_reflector_real(matrix, k)
        H -= 2 * np.outer(v, v @ H)
        H -= 2 * np.outer(H @ v, v)
        Q -= 2 * np.outer(Q @ v, v)
    return H, Q

def define_householder_reflector_complex(matrix: np.ndarray, k: int):
    n = matrix.shape[0]
    x = matrix[k + 1:, k]
    if np.linalg.norm(x[1:]) < 1e-12:
        return np.zeros(n, dtype=complex)
    x0_abs = np.abs(x[0])
    phase = x[0] / x0_abs if x0_abs > 0 else 1.0 + 0.0j
    alpha = -phase * np.linalg.norm(x)
    u = x.copy()
    u[0] = x[0] - alpha
    v_sub = u / np.linalg.norm(u)
    v = np.zeros(n, dtype=complex)
    v[k + 1:] = v_sub
    return v


def matrix_to_hess_complex(matrix: np.ndarray):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    elif matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")
    n = matrix.shape[0]
    H = matrix.astype(complex).copy()
    Q = np.eye(n, dtype=complex) # Initialize Q as the identity matrix
    for k in range(n - 2):
        v = define_householder_reflector_complex(matrix, k)
        v_conj = np.conjugate(v)
        H -= 2 * np.outer(v, v_conj @ H)
        H -= 2 * np.outer(H @ v, v_conj)
        Q -= 2 * np.outer(Q @ v, v_conj)
    return H, Q




print(matrix_to_hess_complex(np.array([[4.0, 1.0, -1.0,5.0], [2.0, 7.0, 1.0,3.0], [1.0, -3.0, 12.0,1.0], [3.0, 2.0, 2.0,1.0]], dtype=complex)))
print(hessqr(np.array([[4.0, 1.0, -1.0,5.0], [2.0, 7.0, 1.0,3.0], [0.0, -3.0, 12.0,1.0], [0.0, 0.0, 2.0,1.0]])))