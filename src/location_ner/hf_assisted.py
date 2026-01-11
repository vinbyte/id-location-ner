"""HuggingFace-assisted heuristics.

This module contains optional logic that *uses a pretrained NER model* to improve
precision/recall when matching a large national gazetteer.

Design goals
------------
- Keep the core extractor (PhraseMatcher + fuzzy) independent of transformers.
- Provide small, composable helpers that an API can call explicitly.

Key feature implemented here
----------------------------
"Subdistrict before Kecamatan" extraction.

Many Indonesian address strings omit the "Kelurahan/Desa" prefix and use a
pattern like:

    "..., Sumber Rejo, Kecamatan Pakal, Surabaya, Jawa Timur ..."

If your gazetteer uses a different surface form (e.g., "Sumberejo"), exact
matching will fail. We can recover it by:
1) Using the NER model to confirm the span is a location-like entity.
2) Linking the district from "Kecamatan <district>".
3) Fuzzy-matching the preceding span as a subdistrict within that district.

This is intentionally conservative (high precision): we only add a subdistrict
mention when we can constrain by a district code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .extractor import LocationMention
from .gazetteer import Gazetteer
from .normalization import normalize_for_lookup


@dataclass(frozen=True)
class NerSpan:
    """A minimal NER location span used for gating."""

    start_char: int
    end_char: int
    score: float


_KECAMATAN_RE = re.compile(
    r"\b(kecamatan|kec\.?)\s+([^,.;\n]{2,60})",
    flags=re.IGNORECASE,
)


def build_ner_spans(
    hf_entities: Iterable[object],
    *,
    min_score: float,
) -> list[NerSpan]:
    """Convert HuggingFace entity objects to an internal span format.

    The CLI passes a list of `HfEntity` objects, but we keep this generic by
    accessing attributes via `getattr`.

    Args:
        hf_entities: Iterable of objects with `start_char`, `end_char`, `score`.
        min_score: Minimum confidence to include the span.

    Returns:
        List of NerSpan.
    """
    spans: list[NerSpan] = []
    for e in hf_entities:
        start = int(getattr(e, "start_char"))
        end = int(getattr(e, "end_char"))
        score = float(getattr(e, "score"))
        if score < min_score:
            continue
        spans.append(NerSpan(start_char=start, end_char=end, score=score))

    spans.sort(key=lambda s: (s.start_char, s.end_char))
    return spans


def add_subdistrict_before_kecamatan(
    text: str,
    *,
    gazetteer: Gazetteer,
    mentions: list[LocationMention],
    ner_spans: list[NerSpan],
    fuzzy_threshold: float,
) -> list[LocationMention]:
    """Try to add a subdistrict mention appearing before a Kecamatan phrase.

    Args:
        text: Original text.
        gazetteer: Loaded Gazetteer.
        mentions: Existing mentions extracted by the base extractor.
        ner_spans: NER spans (LOC/GPE) used as a semantic gate.
        fuzzy_threshold: Minimum fuzzy score (0..100).

    Returns:
        New LocationMention objects (may be empty).
    """
    if fuzzy_threshold < 0 or fuzzy_threshold > 100:
        raise ValueError("fuzzy_threshold must be in range [0, 100]")

    existing = [(m.start_char, m.end_char) for m in mentions]
    added: list[LocationMention] = []

    for m in _KECAMATAN_RE.finditer(text):
        kecamatan_start = m.start(0)
        district_raw = m.group(2).strip()
        if not district_raw:
            continue

        district_code = _infer_district_code(
            gazetteer,
            district_raw=district_raw,
            mentions=mentions,
        )
        if not district_code:
            # We only add a subdistrict mention when district is known.
            continue

        # The "subdistrict-like" candidate span is the comma-separated segment
        # immediately before the word "Kecamatan".
        seg_start, seg_end = _segment_before_index(text, kecamatan_start)
        if seg_start is None:
            continue

        # Intersect with the best NER span inside the segment.
        best_ner = _best_overlapping_ner_span(seg_start, seg_end, ner_spans)
        if best_ner is None:
            continue

        if _overlaps_any((best_ner.start_char, best_ner.end_char), existing):
            continue

        query_text = text[best_ner.start_char : best_ner.end_char].strip()
        query_norm = normalize_for_lookup(query_text)
        if not query_norm:
            continue

        candidates, score, matched_key = gazetteer.fuzzy_match(
            "subdistrict",
            query_norm,
            threshold=fuzzy_threshold,
            district_code=district_code,
        )
        if not candidates:
            continue

        added.append(
            LocationMention(
                level="subdistrict",
                matched_text=query_text,
                start_char=best_ner.start_char,
                end_char=best_ner.end_char,
                candidates=_candidate_dicts_for_mentions(candidates, fuzzy_score=score),
                match_method="fuzzy",
                fuzzy_match_score=score,
                level_hint="subdistrict",
            )
        )

    return added


def _segment_before_index(text: str, index: int) -> tuple[int, int] | tuple[None, None]:
    """Return a (start, end) for the segment before `index`.

    We take the last comma-delimited segment before the given index.
    """
    left = text.rfind(",", 0, index)
    if left == -1:
        return None, None

    start = left + 1
    end = index

    # Trim whitespace.
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    # Drop trailing punctuation.
    while end > start and text[end - 1] in {",", ";", "."}:
        end -= 1
    while end > start and text[end - 1].isspace():
        end -= 1

    if end - start < 2:
        return None, None

    return start, end


def _best_overlapping_ner_span(
    start: int,
    end: int,
    ner_spans: list[NerSpan],
) -> NerSpan | None:
    """Pick the best NER span overlapping the given window."""
    best: NerSpan | None = None
    best_score = -1.0

    for s in ner_spans:
        overlap = _overlap_len((start, end), (s.start_char, s.end_char))
        if overlap <= 0:
            continue

        # Prefer higher overlap, then higher model confidence.
        score = float(overlap) + 0.1 * s.score
        if score > best_score:
            best = s
            best_score = score

    return best


def _infer_district_code(
    gazetteer: Gazetteer,
    *,
    district_raw: str,
    mentions: list[LocationMention],
) -> str | None:
    """Infer the most likely district code for a "Kecamatan <name>" phrase."""
    district_norm = normalize_for_lookup(district_raw)
    if not district_norm:
        return None

    # First try exact district lookup.
    candidates = gazetteer.get_candidates_at_level("district", district_norm)

    # If ambiguous, prefer candidates that match the strongest city/province evidence.
    if candidates and len(candidates) > 1:
        province_code = _best_code_vote(mentions, "province_code")
        city_code = _best_code_vote(mentions, "city_code")

        filtered = [
            c for c in candidates if _match_code(c, "province_code", province_code)
        ]
        if city_code:
            filtered_city = [
                c for c in filtered if _match_code(c, "city_code", city_code)
            ]
            if filtered_city:
                filtered = filtered_city

        if filtered:
            candidates = filtered

    # If still none, try fuzzy match (unconstrained here; constraint will be applied later).
    if not candidates:
        candidates, _score, _key = gazetteer.fuzzy_match(
            "district",
            district_norm,
            threshold=90.0,
            city_code=_best_code_vote(mentions, "city_code"),
        )

    if not candidates:
        return None

    # If multiple remain, pick the one with the most specific code available.
    # District candidates should always have district_code.
    district_codes = {
        c.location.district_code for c in candidates if c.location.district_code
    }
    if len(district_codes) == 1:
        return next(iter(district_codes))

    # Fallback: choose the first candidate deterministically.
    return candidates[0].location.district_code


def _best_code_vote(mentions: list[LocationMention], code_field: str) -> str | None:
    """Return the most likely code by weighted voting over mentions."""
    scores: dict[str, float] = {}

    for m in mentions:
        candidates = m.candidates or []
        n = max(1, len(candidates))

        # Ambiguous mentions contribute less.
        mention_weight = 1.0 / (n**0.5)

        for c in candidates:
            v = c.get(code_field)
            if not isinstance(v, str) or not v.strip():
                continue

            level = c.get("level")
            level_weight = {
                "province": 1.0,
                "city": 2.0,
                "district": 3.0,
                "subdistrict": 4.0,
            }.get(str(level), 1.0)

            scores[v.strip()] = (
                scores.get(v.strip(), 0.0) + mention_weight * level_weight
            )

    if not scores:
        return None

    return max(scores.items(), key=lambda kv: kv[1])[0]


def _match_code(entry: object, field: str, code: str | None) -> bool:
    if code is None:
        return True
    loc = getattr(entry, "location", None)
    if loc is None:
        return True
    v = getattr(loc, field, None)
    return v == code


def _candidate_dicts_for_mentions(
    candidates: list[object],
    *,
    fuzzy_score: float | None,
) -> list[dict[str, object]]:
    """Convert GazetteerEntry objects to the same candidate dict format."""
    out: list[dict[str, object]] = []
    for c in candidates:
        loc = getattr(c, "location")
        d: dict[str, object] = {
            "level": getattr(c, "level"),
            "canonical_name": getattr(c, "canonical_name"),
            "province_name": loc.province_name,
            "province_code": loc.province_code,
            "city_name": loc.city_name,
            "city_code": loc.city_code,
            "district_name": loc.district_name,
            "district_code": loc.district_code,
            "subdistrict_name": loc.subdistrict_name,
            "subdistrict_code": loc.subdistrict_code,
            "postal_code": loc.postal_code,
        }
        if fuzzy_score is not None:
            d["fuzzy_score"] = fuzzy_score
        out.append(d)
    return out


def _overlaps_any(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    s1, e1 = span
    for s2, e2 in others:
        if s1 < e2 and s2 < e1:
            return True
    return False


def _overlap_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0, e - s)
