# Location NER (Indonesia) — Gazetteer + NER Assisted

This repo extracts Indonesian administrative locations (province / city / district / subdistrict) from free text **without retraining** a model.

Core idea:
- Use a **Kemendagri-style gazetteer (CSV)** for structured, code-based results.
- Use **spaCy PhraseMatcher** for fast exact matching.
- Use **optional HuggingFace NER** as a semantic helper (gating + linking), especially for messy/long text.

## Features
- Gazetteer loader that deduplicates by admin codes (prevents duplicate candidates).
- Exact matching with `spaCy` `PhraseMatcher` (case-insensitive).
- Typo-tolerant matching (`--fuzzy`) using `rapidfuzz`.
- Context-aware disambiguation for common prefixes (e.g., `Kel.` vs `Kec.`).
- NER-gated matching (`--ner-filter`) to reduce false positives with national gazetteers.
- HF-assisted recovery for addresses that omit `Kelurahan/Desa` (e.g., "..., Sumber Rejo, Kecamatan Pakal, ...").
- Robust HuggingFace handling for very long texts via chunked inference (no 512-token crash).
- `final_result`: resolves ambiguous mentions into a single best administrative path (conservative: doesn’t force uncertain subdistrict).

## Why gazetteer/phrase-matcher?
- Fast to iterate: update the CSV → results update immediately
- High precision for known locations
- No costly retraining cycle

Trade-off: recall depends on how complete your location list is and how many spelling variants you handle.

## Dataset format
Example CSV: `gazetteer/csv/location_kemendagri_2025.csv`
This location is collected from : https://github.com/cahyadsn/wilayah (thanks bro)

Required columns:
- `province_name`, `province_code`
- `city_name`, `city_code`
- `district_name`, `district_code`
- `subdistrict_name`, `subdistrict_code`
- (optional) `postal_code`

## Install

Requirements: Python 3.10+

### Option A: editable install (recommended)
```bash
python3 -m pip install -U pip
python3 -m pip install -e .
```

### Optional: HuggingFace NER support
```bash
python3 -m pip install -e ".[hf]"
```

## Dev shortcuts
- `make smoke` runs a quick gazetteer-only check.
- `make smoke-hf` runs a quick HF + fuzzy + NER-gate check.

## Quick start

### 1) Gazetteer-only extraction (no transformer)
```bash
location-ner \
  --csv gazetteer/csv/location_kemendagri_2025.csv \
  --text "Alamat. Perum Prana Estate Blok A5/1 Kel. Cikole Kec. Cikole, Kota Sukabumi, Jawa Barat 43115."
```

### 2) HF-assisted extraction (recommended for messy / long text)
This runs HuggingFace NER + linking:

```bash
location-ner \
  --csv gazetteer/csv/location_kemendagri_2025.csv \
  --hf-model cahya/bert-base-indonesian-NER \
  --text "... Kecamatan Pakal, Surabaya, Jawa Timur ..."
```

### 3) Typo-tolerant matching
```bash
location-ner \
  --csv gazetteer/csv/location_kemendagri_2025.csv \
  --fuzzy --fuzzy-threshold 90 \
  --text "... Kelurahan Pangongngan, Kecamatan Manguharjo, Kota Mediun ..."
```

### 4) NER-gated gazetteer matching (recommended for national gazetteers)
With a full Indonesia gazetteer, many short/common words can be valid location names
somewhere (e.g., "Raya"), causing false positives.

`--ner-filter` uses the HuggingFace NER model as a semantic gate: exact PhraseMatcher
matches are kept only if they overlap a model-predicted location span (LOC/GPE).

```bash
location-ner \
  --csv gazetteer/csv/location_kemendagri_2025.csv \
  --hf-model cahya/bert-base-indonesian-NER \
  --ner-filter --ner-filter-min-score 0.5 \
  --text "... JAWA BARAT, KOTA SUKABUMI, Cikole, Cisarua ..."
```

### 5) Compact output (one line JSON)
```bash
location-ner --csv gazetteer/csv/location_kemendagri_2025.csv --text "..." --jsonl
```

## Output
The CLI prints JSON with:
- `gazetteer_mentions`: all matched spans (can be ambiguous)
- `final_result`: a single best administrative path (API-friendly)
- `hf_location_mentions`: only when `--hf-model` is enabled

`final_result` shape:
- `province_name`, `province_code`
- `city_name`, `city_code`
- `district_name`, `district_code`
- `subdistrict_name`, `subdistrict_code`

Resolver behavior:
- Conservative: deeper levels (especially subdistrict) are included only when supported.

Mention fields:
- `match_method`: `exact|fuzzy`
- `fuzzy_match_score`: only set for fuzzy matches
- `level_hint`: may be inferred from text prefixes (e.g., `Kel.` vs `Kec.`)

## Notes / Limitations
- Ambiguity: some names exist in multiple regions; the extractor returns multiple candidates.
- Variants: the matcher includes a few basic normalization variants (case, punctuation, hyphens). If your real data has many aliases, add a synonym layer.
- For very large gazetteers (Indonesia full), PhraseMatcher can still work but you should profile memory/time; in some cases, a trie-based matcher or a database-backed lookup is better.

## Examples
- FastAPI server example: `examples/api_server_fastapi/README.md`

## Project layout
- `src/location_ner/gazetteer.py`: CSV loader + normalized lookup (+ fuzzy indexes)
- `src/location_ner/extractor.py`: spaCy PhraseMatcher extractor (+ fuzzy + context hints)
- `src/location_ner/hf_ner.py`: HuggingFace NER wrapper (chunked for long text)
- `src/location_ner/hf_assisted.py`: HF-assisted heuristics (pre-"Kecamatan" subdistrict)
- `src/location_ner/resolver.py`: final result resolver
- `src/location_ner/cli.py`: command-line interface
