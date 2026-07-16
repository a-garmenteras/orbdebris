import numpy as np


def test_numpy_linear_algebra():
    # Rotating (1, 0) by 90 degrees should land on (0, 1): a cheap sanity
    # check that the numerics stack (BLAS/LAPACK via numpy) is wired up correctly.
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    result = rotation @ np.array([1.0, 0.0])
    assert np.allclose(result, [0.0, 1.0])
