"""twin CLI entry point. Subcommands land phase by phase; see PLAN.md §3.4."""

from importlib.metadata import version as _pkg_version

import typer

app = typer.Typer()


@app.callback()
def main() -> None:
    """twin — personal digital-twin agent framework."""


@app.command()
def version() -> None:
    """Print the installed twin package version."""
    typer.echo(_pkg_version("twin"))


if __name__ == "__main__":
    app()
