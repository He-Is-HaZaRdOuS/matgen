import os

import click

from matrixgen import load_matrix, resize_matrix, save_matrix
from matrixgen.core import RESIZE_METHODS


@click.group()
@click.version_option()
def main():
    """MatrixGen - Sparse matrix generator and resizer CLI."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--size",
    "-s",
    type=click.IntRange(min=1),
    required=True,
    help="New matrix dimension (NxN). Must be >= 1.",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(list(RESIZE_METHODS.keys()), case_sensitive=False),
    default="nearest",
    help="Resizing method to use.",
)
def resize(input_file, output_file, size, method):
    """Resize a sparse matrix to a new size using the specified method."""
    matrix = load_matrix(input_file)

    if matrix is None:
        raise TypeError(f"Failed to load {input_file}")

    try:
        resized = resize_matrix(matrix, new_size=size, method=method)
    except Exception as e:
        raise click.ClickException(str(e))

    file_name = os.path.basename(output_file)
    folder_path = os.path.dirname(output_file) or "."

    save_matrix(resized, file_name, folder_path)


if __name__ == "__main__":
    main()
