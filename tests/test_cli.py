"""Test cases for the __main__ module."""

import os

import pytest
from click.testing import CliRunner

from matrixgen.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_resize_cli(tmp_path):
    import os
    import tempfile

    from scipy.sparse import identity

    from matrixgen.utils.io import load_matrix, save_matrix

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
                "nearest",
            ],
        )

        print("stdout:", result.stdout)
        print("exception:", result.exception)
        assert result.exit_code == 0
        assert output_path.exists()
        resized = load_matrix(output_path)
        assert resized.shape == (20, 20)


def test_resize_cli_missing_input_file(runner, tmp_path):
    output_path = tmp_path / "output.mtx"
    result = runner.invoke(
        main,
        [
            "resize",
            "non_existent.mtx",
            str(output_path),
            "--size",
            "20",
            "--method",
            "nearest",
        ],
    )
    assert result.exit_code != 0
    assert "Error" in result.output or "No such file" in result.output


def test_resize_cli_invalid_method(runner, tmp_path):
    import tempfile

    from scipy.sparse import identity

    from matrixgen.utils.io import save_matrix

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mtx")
        save_matrix(identity(10, format="csr"), "input.mtx", tmpdir)
        output_path = tmp_path / "output.mtx"

        result = runner.invoke(
            main,
            [
                "resize",
                str(input_path),
                str(output_path),
                "--size",
                "20",
                "--method",
                "nonexistent_method",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value for '--method'" in result.output


def test_resize_cli_invalid_size(runner, tmp_path):
    import tempfile

    from scipy.sparse import identity

    from matrixgen.utils.io import save_matrix

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mtx")
        save_matrix(identity(10, format="csr"), "input.mtx", tmpdir)
        output_path = tmp_path / "output.mtx"

        result = runner.invoke(
            main,
            [
                "resize",
                str(input_path),
                str(output_path),
                "--size",
                "-5",
                "--method",
                "nearest",
            ],
        )

        assert result.exit_code != 0
        assert (
            "Invalid value for '--size'" in result.output
            or "Invalid size" in result.output
        )


def test_resize_cli_default_method(runner, tmp_path):
    import tempfile

    from scipy.sparse import identity

    from matrixgen.utils.io import load_matrix, save_matrix

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mtx")
        save_matrix(identity(10, format="csr"), "input.mtx", tmpdir)
        output_path = tmp_path / "output.mtx"

        result = runner.invoke(
            main,
            [
                "resize",
                str(input_path),
                str(output_path),
                "--size",
                "15",
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()
        resized = load_matrix(output_path)
        assert resized.shape == (15, 15)


def test_resize_cli_help(runner):
    result = runner.invoke(main, ["resize", "--help"])
    assert result.exit_code == 0
    assert "Resize a sparse matrix" in result.output
    assert "--size" in result.output
    assert "--method" in result.output
