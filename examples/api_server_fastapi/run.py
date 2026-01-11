"""Run the FastAPI server using env-based configuration.

Why this wrapper?
-----------------
`uvicorn` CLI flags are great, but in many teams it is more convenient to keep
runtime configuration in a `.env` file.

This script reads:
- `LOCATION_NER_API_HOST` (default: 0.0.0.0)
- `LOCATION_NER_API_PORT` (default: 8000)

It then starts Uvicorn with the already-configured `app`.

Usage
-----
python examples/api_server_fastapi/run.py
"""

from __future__ import annotations

import os

import uvicorn

# Importing `main` loads `.env` early (see main.py).
# Import locally so this works when running from inside this folder.
from main import app  # type: ignore  # noqa: E402


def _get_host() -> str:
    return os.getenv("LOCATION_NER_API_HOST", "0.0.0.0").strip() or "0.0.0.0"


def _get_port() -> int:
    raw = os.getenv("LOCATION_NER_API_PORT", "8000").strip()
    try:
        port = int(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid LOCATION_NER_API_PORT={raw!r}") from e

    if not (1 <= port <= 65535):
        raise RuntimeError("LOCATION_NER_API_PORT must be in range [1, 65535]")

    return port


if __name__ == "__main__":
    uvicorn.run(app, host=_get_host(), port=_get_port())
