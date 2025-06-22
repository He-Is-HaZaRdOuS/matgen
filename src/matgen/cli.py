import os

import click

from matgen import load_matrix, resize_matrix, save_matrix
from matgen.generators.bilinear import scale_sparse_matrix_bilinear
from matgen.generators.dct import scale_sparse_matrix_dct_blockwise
from matgen.generators.fourier import scale_sparse_matrix_fourier
from matgen.generators.gaussian import scale_sparse_matrix_gaussian
from matgen.generators.graph import scale_sparse_matrix_graph
from matgen.generators.image import scale_sparse_matrix_image
from matgen.generators.lanczos import scale_sparse_matrix_lanczos
from matgen.generators.nearest import scale_sparse_matrix_nearest
from matgen.generators.wavelet import scale_sparse_matrix_wavelet

RESIZE_METHODS = {
    "bilinear": scale_sparse_matrix_bilinear,
    "dct": scale_sparse_matrix_dct_blockwise,
    "dft": scale_sparse_matrix_fourier,
    "gaussian": scale_sparse_matrix_gaussian,
    "graph": scale_sparse_matrix_graph,
    "image": scale_sparse_matrix_image,
    "lanczos": scale_sparse_matrix_lanczos,
    "nearest-neighbour": scale_sparse_matrix_nearest,
    "wavelet": scale_sparse_matrix_wavelet,
}


@click.group()
@click.version_option()
def main():
    """MatGen - Sparse matrix generator and resizer CLI."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--size", "-s", type=int, required=True, help="New matrix dimension (NxN)."
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(list(RESIZE_METHODS.keys()), case_sensitive=False),
    default="nearest-neighbour",
    help="Resizing method to use.",
)
def resize(input_file, output_file, size, method):
    """Resize a sparse matrix to a new size using the specified method."""
    matrix = load_matrix(input_file)

    resize_fn = RESIZE_METHODS[method.lower()]
    resized = resize_matrix(matrix, new_size=size, method=resize_fn)

    file_name = os.path.basename(output_file)
    folder_path = os.path.dirname(output_file) or "."

    save_matrix(resized, file_name, folder_path)


if __name__ == "__main__":
    main()
