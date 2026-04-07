"""
server/app.py — Content Moderation OpenEnv
===========================================
Entry point for the FastAPI server. Required by openenv validate.
Re-exports the app from the root server module.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export the fully configured app from server.py
from server import app  # noqa: F401  (imported for uvicorn)


def main() -> None:
    """Entry point for project.scripts — required by openenv validate."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
