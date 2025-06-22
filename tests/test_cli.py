"""Test cases for the __main__ module."""

import pytest
from click.testing import CliRunner

from matgen.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_resize_cli(tmp_path):
    import os
    import tempfile

    from scipy.sparse import identity

    from matgen.utils.io import load_matrix, save_matrix

    original = identity(10, format="csr")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mtx")
        output_path = tmp_path / "output.mtx"
        save_matrix(original, "input.mtx", tmpdir)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "resize",
                str(input_path),
                str(output_path),
                "--size",
                "20",
                "--method",
                "nearest-neighbour",
            ],
        )

        print("stdout:", result.stdout)
        print("exception:", result.exception)
        assert result.exit_code == 0
        assert output_path.exists()
        resized = load_matrix(output_path)
        assert resized.shape == (20, 20)
