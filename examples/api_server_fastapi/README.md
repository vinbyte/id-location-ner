# Example API Server (FastAPI)

This is a minimal FastAPI server example for the `location-ner` library.

- Endpoint: `POST /extract`
- Request JSON: `{"text": "..."}`
- Response JSON: `{"text": "...", "final_result": {...}}`

The server loads the gazetteer CSV + HuggingFace NER model **once at startup**
and reuses them for all requests.

## Setup

### 1) Create a virtual environment (recommended)
From `examples/api_server_fastapi/`:

```bash
# IMPORTANT: use Python 3.10–3.12. spaCy does not support Python 3.13 yet.
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

If you don't have `python3.12`, use `python3.11` (or any Python in 3.10–3.12).

### 2) Install dependencies

Important: `location-ner` is the *repo root* project, not the `examples/` folder.

Since this folder is an example, you still need to install the main library
project from the repo root.

If your current working directory is `examples/api_server_fastapi/`:
```bash
python3 -m pip install -e "../..[hf]"
python3 -m pip install -r requirements.txt

# Quick sanity check
python -c "import location_ner; print(location_ner.__version__ if hasattr(location_ner,'__version__') else 'ok')"
```

(If you prefer running from the repo root instead, install with `pip install -e ".[hf]"`
then `pip install -r examples/api_server_fastapi/requirements.txt`.)

### 3) Configure environment

Copy the example env file and edit it:

```bash
cp .env.example .env
```

Then set at least:
- `LOCATION_NER_CSV` (absolute path to your Kemendagri-style CSV)

Optional:
- `LOCATION_NER_API_HOST` (default: `0.0.0.0`)
- `LOCATION_NER_API_PORT` (default: `8000`)
- `LOCATION_NER_HF_MODEL` (default: `cahya/bert-base-indonesian-NER`)
- `LOCATION_NER_FUZZY_THRESHOLD` (default: `90`)
- `LOCATION_NER_NER_FILTER_MIN_SCORE` (default: `0.5`)

Notes:
- `.env` is ignored by git.
- If you want to load a different env file, set `LOCATION_NER_ENV_FILE=/path/to/file.env`.

## Run

### Option A (recommended): run via env-configured wrapper
This reads `LOCATION_NER_API_HOST` and `LOCATION_NER_API_PORT` from your `.env`.

```bash
python run.py
```

### Option B: run uvicorn directly
From this folder:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Test

```bash
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"text":"Graha Suryanata berlokasi di ... Kecamatan Pakal, Surabaya, Jawa Timur 60192"}'
```

## Response

`final_result` is either:
- a single best administrative path (province/city/district/subdistrict + codes), or
- `null` if nothing can be resolved.

The resolver is conservative: it prefers leaving deeper levels empty rather than
forcing an incorrect best-guess.

## Example Test

### Screenshots

#### 1) Subdistrict First
![Subdistrict First](docs/images/subdistrict-first.png)

#### 2) Province First
![Province First](docs/images/province-first.png)

#### 3) Complex text example
![Complex text](docs/images/complex.png)

