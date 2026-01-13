"""Command-line interface for the location extractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extractor import LocationExtractor


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def cli() -> None:
    """Console-script entry point.

    This exists so you can run the tool as a normal command after installing the
    package into a virtual environment:

        location-ner --csv ... --text "..."

    Keeping this wrapper separate also avoids relying on `python -m ...`.
    """
    raise SystemExit(main())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Indonesian administrative locations using a CSV gazetteer "
            "(spaCy PhraseMatcher) with optional HuggingFace NER fallback."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the location CSV (Kemendagri-like format).",
    )

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Input text to analyze.")
    text_group.add_argument("--text-file", help="Read input text from a file.")

    parser.add_argument(
        "--max-level",
        choices=["province", "city", "district", "subdistrict"],
        default=None,
        help="Limit matching to a maximum administrative level.",
    )

    parser.add_argument(
        "--hf-model",
        default=None,
        help=(
            "Optional HuggingFace NER model name. Example: cahya/bert-base-indonesian-NER. "
            "If provided, the CLI will also return model-predicted location entities."
        ),
    )

    parser.add_argument(
        "--ner-filter",
        action="store_true",
        help=(
            "Use the HuggingFace NER model as a semantic gate for gazetteer matching. "
            "When enabled, exact PhraseMatcher matches are kept only if they overlap a "
            "NER-predicted location span (LOC/GPE). Requires --hf-model."
        ),
    )
    parser.add_argument(
        "--ner-filter-min-score",
        type=float,
        default=0.5,
        help="Minimum NER entity score to be used as a filter span. Default: 0.5.",
    )

    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help=(
            "Enable typo-tolerant matching. This adds a second pass that tries to fuzzy-match "
            "admin-prefixed spans like 'Kelurahan X'/'Kecamatan Y'/'Kota Z'."
        ),
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=90.0,
        help="Minimum fuzzy score (0..100). Higher is stricter. Default: 90.",
    )

    parser.add_argument(
        "--jsonl",
        "--compact-json",
        action="store_true",
        help=(
            "Print output as a single-line JSON (JSONL-style). Useful for logging or large batches."
        ),
    )

    args = parser.parse_args(argv)

    text = args.text
    if text is None:
        text = _read_text_file(Path(args.text_file))

    extractor = LocationExtractor.from_csv(args.csv, max_level=args.max_level)

    hf = None
    hf_locations = None
    allowed_char_ranges = None

    # If a HuggingFace model is provided, run it once and reuse the results.
    ner_spans = None
    if args.hf_model:
        try:
            from .hf_ner import HuggingFaceNer
            from .hf_assisted import build_ner_spans

            hf = HuggingFaceNer(args.hf_model)
            hf_locations = hf.extract_locations(text)

            if args.ner_filter_min_score < 0.0 or args.ner_filter_min_score > 1.0:
                raise SystemExit("--ner-filter-min-score must be in range [0, 1]")

            ner_spans = build_ner_spans(
                hf_locations, min_score=args.ner_filter_min_score
            )

            if args.ner_filter:
                allowed_char_ranges = [(s.start_char, s.end_char) for s in ner_spans]
        except ModuleNotFoundError as e:
            raise SystemExit(
                "transformers/torch is not installed. Install with: pip install -e '.[hf]'"
            ) from e
    elif args.ner_filter:
        raise SystemExit("--ner-filter requires --hf-model")

    mentions = extractor.extract(
        text,
        fuzzy=args.fuzzy,
        fuzzy_threshold=args.fuzzy_threshold,
        allowed_char_ranges=allowed_char_ranges,
    )

    # HuggingFace-assisted recovery: infer subdistrict right before "Kecamatan".
    # This helps for cases like "Sumber Rejo, Kecamatan Pakal" where the subdistrict
    # has no explicit prefix (Kelurahan/Desa) and may contain spacing variations.
    if hf_locations is not None and ner_spans is not None and args.fuzzy:
        from .hf_assisted import add_subdistrict_before_kecamatan

        added = add_subdistrict_before_kecamatan(
            text,
            gazetteer=extractor.gazetteer,
            mentions=mentions,
            ner_spans=ner_spans,
            fuzzy_threshold=args.fuzzy_threshold,
        )
        if added:
            mentions.extend(added)
            mentions.sort(key=lambda m: (m.start_char, m.end_char))

    gazetteer_mentions = [m.__dict__ for m in mentions]

    # Resolve a single best administrative path for API-friendly output.
    from .resolver import prune_mentions_to_resolved, resolve_best_location

    resolved = resolve_best_location(gazetteer_mentions)
    pruned_mentions = prune_mentions_to_resolved(gazetteer_mentions, resolved)

    out: dict[str, object] = {
        "text": text,
        "gazetteer_mentions": pruned_mentions,
        "final_result": resolved.to_dict() if resolved else None,
    }

    # Optional HuggingFace NER extraction.
    # NOTE: the model (if provided) was already executed above, and `hf_locations` is reused.
    if args.hf_model:
        assert hf_locations is not None

        # Lightweight linking: normalize the model span and do an exact gazetteer lookup.
        # If you need fuzzy matching / aliases, extend `normalize_for_lookup` or add a synonym layer.
        from .normalization import normalize_for_lookup, strip_admin_prefix

        # Infer unique parent codes from the gazetteer pass to constrain fuzzy linking.
        # We only use a constraint if it is unambiguous.
        province_codes: set[str] = set()
        city_codes: set[str] = set()
        district_codes: set[str] = set()

        for m in mentions:
            for c in m.candidates:
                if (
                    isinstance(c.get("province_code"), str)
                    and c["province_code"].strip()
                ):
                    province_codes.add(c["province_code"].strip())
                if isinstance(c.get("city_code"), str) and c["city_code"].strip():
                    city_codes.add(c["city_code"].strip())
                if (
                    isinstance(c.get("district_code"), str)
                    and c["district_code"].strip()
                ):
                    district_codes.add(c["district_code"].strip())

        province_code = next(iter(province_codes)) if len(province_codes) == 1 else None
        city_code = next(iter(city_codes)) if len(city_codes) == 1 else None
        district_code = next(iter(district_codes)) if len(district_codes) == 1 else None

        linked: list[dict[str, object]] = []
        for ent in hf_locations:
            normalized = normalize_for_lookup(ent.text)
            candidates = extractor.gazetteer.get_candidates(normalized)

            link_method = "exact"
            link_score = None
            matched_key = None

            # If the model span contains an admin prefix (e.g., "kelurahan X"),
            # strip it and try again.
            stripped, hint = strip_admin_prefix(ent.text)
            normalized_stripped = None
            level_hint = None

            if not candidates and stripped and stripped != normalized:
                normalized_stripped = stripped
                level_hint = hint
                candidates = extractor.gazetteer.get_candidates(stripped)
                link_method = "stripped_exact"

            # If we have a strong hint about the admin level, filter candidates.
            if level_hint and candidates:
                filtered = [c for c in candidates if c.level == level_hint]
                if filtered:
                    candidates = filtered

            # Fuzzy linking if enabled and still no candidates.
            if args.fuzzy and not candidates:
                # Decide the most likely admin level for fuzzy search.
                fuzzy_level = None
                query = normalized

                if level_hint:
                    fuzzy_level = level_hint
                    query = normalized_stripped or normalized
                elif (
                    normalized.startswith("kota ")
                    or normalized.startswith("kabupaten ")
                    or normalized.startswith("kab ")
                ):
                    fuzzy_level = "city"
                    query = normalized

                if fuzzy_level:
                    candidates, link_score, matched_key = (
                        extractor.gazetteer.fuzzy_match(
                            fuzzy_level,
                            query,
                            threshold=args.fuzzy_threshold,
                            province_code=province_code
                            if fuzzy_level == "city"
                            else None,
                            city_code=city_code
                            if fuzzy_level in {"district", "subdistrict"}
                            else None,
                            district_code=district_code
                            if fuzzy_level == "subdistrict"
                            else None,
                        )
                    )
                    if candidates:
                        link_method = "fuzzy"

            linked.append(
                {
                    **ent.__dict__,
                    "normalized": normalized,
                    "normalized_stripped": normalized_stripped,
                    "level_hint": level_hint,
                    "link_method": link_method,
                    "link_score": link_score,
                    "matched_key": matched_key,
                    "gazetteer_candidates": [
                        {
                            "level": c.level,
                            "canonical_name": c.canonical_name,
                            "province_name": c.location.province_name,
                            "province_code": c.location.province_code,
                            "city_name": c.location.city_name,
                            "city_code": c.location.city_code,
                            "district_name": c.location.district_name,
                            "district_code": c.location.district_code,
                            "subdistrict_name": c.location.subdistrict_name,
                            "subdistrict_code": c.location.subdistrict_code,
                            "postal_code": c.location.postal_code,
                        }
                        for c in candidates
                    ],
                }
            )

        out["hf_location_mentions"] = linked

    if args.jsonl:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
