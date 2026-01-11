"""Example FastAPI server for `location-ner`.

Goals
-----
- Minimal API response: {"text": ..., "final_result": ...}
- Load gazetteer + HF model once at startup (do NOT reload per request)
- Default behavior:
  - HF model always ON
  - NER-gated gazetteer matching ON
  - Fuzzy matching ON

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
- LOCATION_NER_HF_MODEL (default: cahya/bert-base-indonesian-NER)
- LOCATION_NER_FUZZY_THRESHOLD (default: 90)
- LOCATION_NER_NER_FILTER_MIN_SCORE (default: 0.5)

Run (example)
-------------
1) Install dependencies:
   pip install -e ".[hf]"
   pip install -r examples/api_server_fastapi/requirements.txt

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
    hf_model = os.getenv(
        "LOCATION_NER_HF_MODEL", "cahya/bert-base-indonesian-NER"
    ).strip()

    fuzzy_threshold = _get_env_float("LOCATION_NER_FUZZY_THRESHOLD", 90.0)
    ner_filter_min_score = _get_env_float("LOCATION_NER_NER_FILTER_MIN_SCORE", 0.5)

    if not (0.0 <= ner_filter_min_score <= 1.0):
        raise RuntimeError("LOCATION_NER_NER_FILTER_MIN_SCORE must be in range [0, 1]")

    # Load gazetteer + build matcher.
    extractor = LocationExtractor.from_csv(csv_path)

    # Load HuggingFace model pipeline.
    # This is the most expensive step; keep it global.
    hf_ner = HuggingFaceNer(hf_model)

    STATE.extractor = extractor
    STATE.hf_ner = hf_ner
    STATE.fuzzy_threshold = float(fuzzy_threshold)
    STATE.ner_filter_min_score = float(ner_filter_min_score)

    yield


app = FastAPI(title="Location NER API", version="0.1.0", lifespan=lifespan)


@app.post("/extract", response_model=ExtractResponse)
def extract_locations(req: ExtractRequest) -> ExtractResponse:
    """Extract a single best administrative path from input text."""
    if STATE.extractor is None or STATE.hf_ner is None:
        raise HTTPException(status_code=503, detail="Server is still starting up")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    # 1) Run HF NER once.
    hf_locations = STATE.hf_ner.extract_locations(text)

    # 2) Build NER spans and use them as an allowlist for exact gazetteer matches.
    ner_spans = build_ner_spans(hf_locations, min_score=STATE.ner_filter_min_score)
    allowed_char_ranges = [(s.start_char, s.end_char) for s in ner_spans]

    # 3) Gazetteer extraction (exact + fuzzy) with NER gating.
    mentions = STATE.extractor.extract(
        text,
        fuzzy=True,
        fuzzy_threshold=STATE.fuzzy_threshold,
        allowed_char_ranges=allowed_char_ranges,
    )

    # 4) HF-assisted recovery: try to find subdistrict immediately before "Kecamatan".
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

    # 5) Resolve final result.
    gazetteer_mentions = [asdict(m) for m in mentions]
    resolved = resolve_best_location(gazetteer_mentions)

    return ExtractResponse(
        text=text,
        final_result=resolved.to_dict() if resolved is not None else None,
    )
