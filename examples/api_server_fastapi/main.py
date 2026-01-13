"""Example FastAPI server for `location-ner`.

Goals
-----
- Minimal API response: {"text": ..., "final_result": ...}
- Load gazetteer once at startup (do NOT reload per request)
- Optional: load HF model once at startup when enabled
- Default behavior (when `LOCATION_NER_HF_MODEL` is NOT set):
  - Gazetteer matching ON
  - Fuzzy matching ON
- When `LOCATION_NER_HF_MODEL` is set:
  - HF NER ON
  - NER-gated gazetteer matching ON
  - HF-assisted heuristics ON

Endpoint
--------
POST /extract
Request JSON:
    {"text": "..."}
Response JSON:
    {
      "text": "...",
      "final_result": {
        "province_name": "...",
        "province_code": "...",
        "city_name": "...",
        "city_code": "...",
        "district_name": "...",
        "district_code": "...",
        "subdistrict_name": "...",
        "subdistrict_code": "..."
      }
    }

Required environment variables
------------------------------
- LOCATION_NER_CSV
    Absolute path to your Kemendagri-style CSV.

Optional environment variables
------------------------------
- LOCATION_NER_FUZZY_THRESHOLD (default: 90)

HF/NER-assisted mode (optional)
------------------------------
If you set `LOCATION_NER_HF_MODEL`, the server enables HuggingFace NER + NER-gated matching.

- LOCATION_NER_HF_MODEL
    HuggingFace model name (e.g. cahya/bert-base-indonesian-NER)
- LOCATION_NER_NER_FILTER_MIN_SCORE (default: 0.5)
    Minimum NER entity score used as a gating span.

Run (example)
-------------
1) Install dependencies:
   # From examples/api_server_fastapi/
   python3 -m pip install -r requirements.txt

2) Export env:
   export LOCATION_NER_CSV="/abs/path/to/location_kemendagri_2025.csv"

3) Start server:
   uvicorn examples.api_server_fastapi.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from location_ner import LocationExtractor, resolve_best_location
from location_ner.hf_assisted import add_subdistrict_before_kecamatan, build_ner_spans
from location_ner.hf_ner import HuggingFaceNer


class ExtractRequest(BaseModel):
    """Request payload for POST /extract."""

    text: str = Field(..., min_length=1, description="Input text to analyze.")


class ExtractResponse(BaseModel):
    """Response payload for POST /extract."""

    text: str
    final_result: dict[str, Any] | None


def _load_dotenv_if_present() -> None:
    """Load a local `.env` file if present.

    Behavior:
    - If `LOCATION_NER_ENV_FILE` is set, load that file.
    - Otherwise try:
      1) `.env` in the current working directory
      2) `.env` next to this `main.py`

    Precedence:
    - Real environment variables always win.
    - `.env` only fills missing variables.

    This keeps the example dependency-free (no python-dotenv).
    """

    def parse_line(line: str) -> tuple[str, str] | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        if "=" not in line:
            return None
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Remove simple quotes.
        if (
            value.startswith(('"', "'"))
            and value.endswith(('"', "'"))
            and len(value) >= 2
        ):
            value = value[1:-1]

        if not key:
            return None
        return key, value

    candidates: list[Path] = []
    explicit = os.getenv("LOCATION_NER_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path(__file__).resolve().parent / ".env")

    env_path = next((p for p in candidates if p.exists() and p.is_file()), None)
    if env_path is None:
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_line(raw)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ:
            continue
        os.environ[key] = value


# Load `.env` early so it is visible in lifespan.
_load_dotenv_if_present()


def _get_env_required(name: str) -> str:
    """Read a required environment variable."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _get_env_float(name: str, default: float) -> float:
    """Read an environment variable as float with a default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid float env var {name}={raw!r}") from e


class _AppState:
    """Holds long-lived objects shared across requests."""

    def __init__(self) -> None:
        self.extractor: LocationExtractor | None = None
        self.hf_ner: HuggingFaceNer | None = None
        self.fuzzy_threshold: float = 90.0
        self.ner_filter_min_score: float = 0.5


STATE = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources once and reuse them."""
    csv_path = _get_env_required("LOCATION_NER_CSV")

    fuzzy_threshold = _get_env_float("LOCATION_NER_FUZZY_THRESHOLD", 90.0)

    hf_model_raw = os.getenv("LOCATION_NER_HF_MODEL")
    hf_model = hf_model_raw.strip() if hf_model_raw is not None else ""
    hf_enabled = bool(hf_model)

    ner_filter_min_score = 0.5
    hf_ner: HuggingFaceNer | None = None

    if hf_enabled:
        ner_filter_min_score = _get_env_float("LOCATION_NER_NER_FILTER_MIN_SCORE", 0.5)
        if not (0.0 <= ner_filter_min_score <= 1.0):
            raise RuntimeError(
                "LOCATION_NER_NER_FILTER_MIN_SCORE must be in range [0, 1]"
            )

    # Load gazetteer + build matcher.
    extractor = LocationExtractor.from_csv(csv_path)

    # Load HuggingFace model pipeline only when enabled.
    if hf_enabled:
        try:
            hf_ner = HuggingFaceNer(hf_model)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "HuggingFace mode requires extra deps. Install with: pip install 'id-location-ner[hf]'"
            ) from e

    STATE.extractor = extractor
    STATE.hf_ner = hf_ner
    STATE.fuzzy_threshold = float(fuzzy_threshold)
    STATE.ner_filter_min_score = float(ner_filter_min_score)

    yield


app = FastAPI(title="Location NER API", version="0.1.0", lifespan=lifespan)


@app.post("/extract", response_model=ExtractResponse)
def extract_locations(req: ExtractRequest) -> ExtractResponse:
    """Extract a single best administrative path from input text."""
    if STATE.extractor is None:
        raise HTTPException(status_code=503, detail="Server is still starting up")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    # Default: gazetteer-only, like `make smoke`.
    if STATE.hf_ner is None:
        mentions = STATE.extractor.extract(
            text,
            fuzzy=True,
            fuzzy_threshold=STATE.fuzzy_threshold,
            allowed_char_ranges=None,
        )
        gazetteer_mentions = [asdict(m) for m in mentions]
        resolved = resolve_best_location(gazetteer_mentions)
        return ExtractResponse(
            text=text,
            final_result=resolved.to_dict() if resolved is not None else None,
        )

    # HF/NER-assisted mode.
    hf_locations = STATE.hf_ner.extract_locations(text)

    ner_spans = build_ner_spans(hf_locations, min_score=STATE.ner_filter_min_score)
    allowed_char_ranges = [(s.start_char, s.end_char) for s in ner_spans]

    mentions = STATE.extractor.extract(
        text,
        fuzzy=True,
        fuzzy_threshold=STATE.fuzzy_threshold,
        allowed_char_ranges=allowed_char_ranges,
    )

    added = add_subdistrict_before_kecamatan(
        text,
        gazetteer=STATE.extractor.gazetteer,
        mentions=mentions,
        ner_spans=ner_spans,
        fuzzy_threshold=STATE.fuzzy_threshold,
    )
    if added:
        mentions.extend(added)
        mentions.sort(key=lambda m: (m.start_char, m.end_char))

    gazetteer_mentions = [asdict(m) for m in mentions]
    resolved = resolve_best_location(gazetteer_mentions)

    return ExtractResponse(
        text=text,
        final_result=resolved.to_dict() if resolved is not None else None,
    )
