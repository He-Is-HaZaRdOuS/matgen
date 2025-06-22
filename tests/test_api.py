import numpy as np
import pytest
from scipy.sparse import identity

from matgen import RESIZE_METHODS, resize_matrix
from matgen.utils.io import load_matrix, save_matrix


@pytest.mark.parametrize("method", RESIZE_METHODS)
def test_resize_matrix_correct_size(method):
    original = identity(100, format="csr")
    new_size = 200
    if method == "gaussian":
        with pytest.raises(ValueError):
            resized = resize_matrix(original, new_size=new_size, method=method)
    else:
        resized = resize_matrix(original, new_size=new_size, method=method)
        assert resized.shape == (new_size, new_size), (
            f"{method} failed to resize properly"
        )


def test_save_and_load_matrix(tmp_path):
    matrix = identity(10, format="csr")
    file_name = "test_matrix.mtx"
    folder = tmp_path

    if len(list(RESIZE_METHODS)) == 0:
        exit(0)

    save_matrix(matrix, file_name, str(folder))

    loaded = load_matrix(str(folder / file_name))

    assert loaded.shape == matrix.shape
    assert np.allclose(matrix.toarray(), loaded.toarray())


def test_resize_invalid_input1():
    with pytest.raises((TypeError, ValueError)):
        resize_matrix(
            "not_a_matrix", new_size=10, method=RESIZE_METHODS["bilinear"]["fn"]
        )


def test_resize_invalid_input2():
    with pytest.raises((TypeError, ValueError)):
        original = identity(10, format="csr")
        resize_matrix(
            original, new_size=-1, method=RESIZE_METHODS["bilinear"]["fn"]
        )
