"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """MatGen."""


if __name__ == "__main__":
    main(prog_name="matgen")  # pragma: no cover
