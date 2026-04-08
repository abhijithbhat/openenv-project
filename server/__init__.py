# server/__init__.py
# Re-exports the FastAPI application from app.py so that:
#   - `uvicorn server:app` resolves correctly (the Dockerfile CMD)
#   - `[project.scripts] server = "server:main"` in pyproject.toml resolves correctly
from app import app, main  # noqa: F401  re-export

__all__ = ["app", "main"]
