# server/app.py
# Compatibility shim required by the `openenv validate` CLI tool.
#
# The validator checks that server/app.py:
#   1. Exists at this exact path
#   2. Contains "def main(" (text scan, not AST)
#   3. Contains both "__name__" and "main()" (text scan)
#
# The real FastAPI application lives in app.py at the repository root.
# This file delegates to it — zero duplication of logic.

from app import app  # noqa: F401  — re-export the FastAPI instance


def main() -> None:
    """Entry point delegating to the real application in app.py."""
    from app import main as _real_main
    _real_main()


if __name__ == "__main__":
    main()
