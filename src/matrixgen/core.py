from scipy.sparse import csr_matrix

from matrixgen.generators.bilinear import scale_sparse_matrix_bilinear
from matrixgen.generators.dct import scale_sparse_matrix_dct_blockwise
from matrixgen.generators.fourier import scale_sparse_matrix_fourier
from matrixgen.generators.gaussian import scale_sparse_matrix_gaussian
from matrixgen.generators.graph import scale_sparse_matrix_graph
from matrixgen.generators.image import scale_sparse_matrix_image
from matrixgen.generators.lanczos import scale_sparse_matrix_lanczos
from matrixgen.generators.nearest import scale_sparse_matrix_nearest
from matrixgen.generators.wavelet import scale_sparse_matrix_wavelet

# Mapping from method name to function
RESIZE_METHODS = {
    "bilinear": {
        "fn": scale_sparse_matrix_bilinear,
        "can_upscale": True,
    },
    "dct": {
        "fn": scale_sparse_matrix_dct_blockwise,
        "can_upscale": True,
    },
    "dft": {
        "fn": scale_sparse_matrix_fourier,
        "can_upscale": True,
    },
    "gaussian": {
        "fn": scale_sparse_matrix_gaussian,
        "can_upscale": False,  # Downscale-only
    },
    "graph": {
        "fn": scale_sparse_matrix_graph,
        "can_upscale": True,
    },
    "image": {
        "fn": scale_sparse_matrix_image,
        "can_upscale": True,
    },
    "lanczos": {
        "fn": scale_sparse_matrix_lanczos,
        "can_upscale": True,
    },
    "nearest": {
        "fn": scale_sparse_matrix_nearest,
        "can_upscale": True,
    },
    "wavelet": {
        "fn": scale_sparse_matrix_wavelet,
        "can_upscale": True,
    },
}


def resize_matrix(matrix: csr_matrix, new_size: int, method: str) -> csr_matrix:
    """
    Resize the input sparse matrix using the selected method.

    Args:
        original_matrix: The input sparse matrix (CSR).
        new_size: The desired size (square) for resizing.
        method: The desired resize method from RESIZE_METHODS list.

    Returns:
        A resized CSR matrix.
    """
    if new_size < 1:
        raise ValueError("Size must be positive")

    if method not in RESIZE_METHODS:
        raise ValueError(f"Unknown resize method '{method}'")

    method_info = RESIZE_METHODS[method]
    if new_size > max(matrix.shape) and not method_info["can_upscale"]:
        raise ValueError(f"Method '{method}' cannot be used for upscaling")

    return method_info["fn"](matrix, new_size)
