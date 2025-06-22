from enum import Enum
from typing import Callable, Union

from scipy.sparse import csr_matrix

from matgen.generators.bilinear import scale_sparse_matrix_bilinear
from matgen.generators.dct import scale_sparse_matrix_dct_blockwise
from matgen.generators.fourier import scale_sparse_matrix_fourier
from matgen.generators.gaussian import scale_sparse_matrix_gaussian
from matgen.generators.graph import scale_sparse_matrix_graph
from matgen.generators.image import scale_sparse_matrix_image
from matgen.generators.lanczos import scale_sparse_matrix_lanczos
from matgen.generators.nearest import scale_sparse_matrix_nearest
from matgen.generators.wavelet import scale_sparse_matrix_wavelet


class ResizeMethod(Enum):
    BILINEAR = scale_sparse_matrix_bilinear
    DCT = scale_sparse_matrix_dct_blockwise
    DFT = scale_sparse_matrix_fourier
    GAUSSIAN = scale_sparse_matrix_gaussian
    GRAPH = scale_sparse_matrix_graph
    IMAGE = scale_sparse_matrix_image
    LANCZOS = scale_sparse_matrix_lanczos
    NEAREST_NEIGHBOUR = scale_sparse_matrix_nearest
    WAVELET = scale_sparse_matrix_wavelet


def resize_matrix(
    original_matrix: csr_matrix,
    new_size: int,
    method: Union[
        ResizeMethod, Callable[[csr_matrix, int], csr_matrix]
    ] = ResizeMethod.NEAREST_NEIGHBOUR,
) -> csr_matrix:
    """
    Resize the input sparse matrix using the selected method.

    Args:
        original_matrix: The input sparse matrix (CSR).
        new_size: The desired size (square) for resizing.
        method: Either a ResizeMethod enum or a custom callable function.

    Returns:
        A resized CSR matrix.
    """
    if isinstance(method, ResizeMethod):
        resize_func = method.value
    elif callable(method):
        resize_func = method
    else:
        raise ValueError(
            "Invalid resize method. Must be a predefined ResizeMethod function or a compatible callable."
        )

    return resize_func(original_matrix, new_size)
