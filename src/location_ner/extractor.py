"""Core extraction logic (gazetteer + PhraseMatcher).

This extractor is designed to be:
- Fast: no ML training, just phrase matching
- Deterministic: same input yields same output
- Extensible: add synonym logic or context-based disambiguation later

High-level idea:
1) Load a gazetteer from CSV (province/city/district/subdistrict)
2) Build PhraseMatcher patterns for all gazetteer names (plus simple variants)
3) Match phrases in a given text
4) Resolve matches back to gazetteer records (including codes)

Optional fuzzy matching
----------------------
PhraseMatcher is exact-match. For real-world address texts, user input often
contains typos (e.g., "Mediun" vs "Madiun").

When `fuzzy=True`, we add a second pass that:
- looks for admin-prefixed spans ("Kelurahan X", "Kecamatan Y", "Kota Z")
- fuzzy-matches the extracted name against the gazetteer
- optionally constrains candidates using already-detected parent locations

This keeps fuzzy matching both *useful* and *safe* (avoids random matches).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.util import filter_spans

from .gazetteer import Gazetteer, GazetteerEntry
from .normalization import (
    generate_city_surface_variants,
    generate_surface_variants,
    normalize_for_lookup,
)


@dataclass(frozen=True)
class LocationMention:
    """One matched location mention in text."""

    level: str
    matched_text: str
    start_char: int
    end_char: int
    candidates: list[dict[str, Any]]

    # Metadata fields that make debugging and downstream post-processing easier.
    match_method: str = "exact"  # exact|fuzzy
    fuzzy_match_score: float | None = None
    level_hint: str | None = None


@dataclass(frozen=True)
class _PrefixedSpan:
    """A span extracted via an administrative prefix regex."""

    level_hint: str
    text: str
    start_char: int
    end_char: int


class LocationExtractor:
    """Extracts administrative locations from text using a CSV gazetteer."""

    def __init__(
        self,
        gazetteer: Gazetteer,
        language: str = "id",
        max_level: str | None = None,
    ):
        """Create a LocationExtractor.

        Args:
            gazetteer: Loaded Gazetteer.
            language: spaCy language code. "id" works without downloading a model.
            max_level: Optional constraint to only match up to a given level.
                Example: "district" will include province/city/district but skip subdistrict.
        """
        self.gazetteer = gazetteer
        self.language = language
        self.max_level = max_level

        # We do not need a full pretrained spaCy pipeline.
        # A blank tokenizer is enough for PhraseMatcher.
        self.nlp = spacy.blank(language)

        # PhraseMatcher: efficient exact phrase matching over token attributes.
        # attr="LOWER" makes matching case-insensitive at token level.
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        self._build_patterns()

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        language: str = "id",
        max_level: str | None = None,
    ) -> "LocationExtractor":
        """Convenience constructor: build extractor directly from a CSV."""
        gazetteer = Gazetteer.from_csv(csv_path)
        return cls(gazetteer=gazetteer, language=language, max_level=max_level)

    def extract(
        self,
        text: str,
        *,
        fuzzy: bool = False,
        fuzzy_threshold: float = 90.0,
        allowed_char_ranges: list[tuple[int, int]] | None = None,
    ) -> list[LocationMention]:
        """Extract location mentions from input text.

        Args:
            text: Free text paragraph.
            fuzzy: If True, enables typo-tolerant matching (see module docstring).
            fuzzy_threshold: Minimum fuzzy score (0..100). Higher is stricter.
            allowed_char_ranges: Optional allowlist of character spans.
                If provided, exact PhraseMatcher matches are kept only when they overlap
                at least one allowed range.

                This is useful with large national gazetteers where short common words
                (e.g., "Raya") can be valid locations somewhere but are often not
                locations in address text.

                A common strategy is to generate these ranges using a pretrained NER
                model (LOC/GPE spans) and use them as a "semantic gate" for gazetteer
                matching.

        Returns:
            A list of LocationMention objects (sorted by appearance in text).
        """
        if fuzzy_threshold < 0 or fuzzy_threshold > 100:
            raise ValueError("fuzzy_threshold must be in range [0, 100]")

        mentions = self._extract_exact(text, allowed_char_ranges=allowed_char_ranges)

        if fuzzy:
            mentions = self._add_fuzzy_mentions(
                text,
                mentions=mentions,
                threshold=fuzzy_threshold,
            )

        return sorted(mentions, key=lambda m: (m.start_char, m.end_char))

    def _extract_exact(
        self,
        text: str,
        *,
        allowed_char_ranges: list[tuple[int, int]] | None,
    ) -> list[LocationMention]:
        """Exact phrase matching using spaCy PhraseMatcher."""
        doc = self.nlp(text)
        matches = self.matcher(doc)

        # Convert matcher results into spans.
        spans = [doc[start:end] for _match_id, start, end in matches]

        # Remove overlaps: prefer longer spans.
        spans = filter_spans(spans)
        spans = sorted(spans, key=lambda s: (s.start_char, s.end_char))

        mentions: list[LocationMention] = []
        for span in spans:
            # Optional semantic gating (e.g., NER-based allowlist).
            if allowed_char_ranges is not None and not _overlaps_any(
                (span.start_char, span.end_char), allowed_char_ranges
            ):
                continue

            # Filter obvious false positives early.
            if _is_obvious_non_location(text, span.start_char, span.end_char):
                continue

            normalized = normalize_for_lookup(span.text)
            candidates = self.gazetteer.get_candidates(normalized)

            # If normalization makes it not found (rare), skip.
            if not candidates:
                continue

            # If we can infer an explicit admin prefix from the raw text (e.g., "Kel." / "Kec."),
            # use it as a hint to filter ambiguous candidates.
            level_hint = _infer_level_hint_from_left_context(text, span.start_char)
            if level_hint:
                filtered = [c for c in candidates if c.level == level_hint]
                if filtered:
                    candidates = filtered

            candidate_dicts = _candidate_dicts(candidates)

            # The mention level is the highest-granularity candidate level we matched.
            # If ambiguous, you will see multiple levels in candidates.
            mention_level = _pick_best_level(candidates)

            mentions.append(
                LocationMention(
                    level=mention_level,
                    matched_text=span.text,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    candidates=candidate_dicts,
                    match_method="exact",
                    level_hint=level_hint,
                )
            )

        return mentions

    def _add_fuzzy_mentions(
        self,
        text: str,
        *,
        mentions: list[LocationMention],
        threshold: float,
    ) -> list[LocationMention]:
        """Add fuzzy matches for common address patterns with typos."""
        # Infer unique parent codes from already-found mentions.
        province_code = _infer_unique_code(mentions, "province_code")
        city_code = _infer_unique_code(mentions, "city_code")
        district_code = _infer_unique_code(mentions, "district_code")

        existing_spans = [(m.start_char, m.end_char) for m in mentions]

        for pref in _extract_prefixed_spans(text):
            if _overlaps_any((pref.start_char, pref.end_char), existing_spans):
                continue

            normalized = normalize_for_lookup(pref.text)
            if not normalized:
                continue

            # First try exact candidates at the hinted level.
            candidates = self.gazetteer.get_candidates_at_level(
                pref.level_hint, normalized
            )
            score: float | None = None
            matched_key: str | None = None

            if not candidates:
                # Fuzzy match constrained by the best available parent.
                if pref.level_hint == "city":
                    candidates, score, matched_key = self.gazetteer.fuzzy_match(
                        "city",
                        normalized,
                        threshold=threshold,
                        province_code=province_code,
                    )
                elif pref.level_hint == "district":
                    candidates, score, matched_key = self.gazetteer.fuzzy_match(
                        "district",
                        normalized,
                        threshold=threshold,
                        city_code=city_code,
                    )
                elif pref.level_hint == "subdistrict":
                    candidates, score, matched_key = self.gazetteer.fuzzy_match(
                        "subdistrict",
                        normalized,
                        threshold=threshold,
                        city_code=city_code,
                        district_code=district_code,
                    )
                elif pref.level_hint == "province":
                    candidates, score, matched_key = self.gazetteer.fuzzy_match(
                        "province",
                        normalized,
                        threshold=threshold,
                    )

            if not candidates:
                continue

            candidate_dicts = _candidate_dicts(candidates, fuzzy_score=score)

            mentions.append(
                LocationMention(
                    level=_pick_best_level(candidates),
                    matched_text=text[pref.start_char : pref.end_char],
                    start_char=pref.start_char,
                    end_char=pref.end_char,
                    candidates=candidate_dicts,
                    match_method="fuzzy",
                    fuzzy_match_score=score,
                    level_hint=pref.level_hint,
                )
            )

        return mentions

    def _build_patterns(self) -> None:
        """Build PhraseMatcher patterns from the gazetteer."""
        allowed_levels = _levels_up_to(self.max_level)

        # We add patterns grouped by a stable key.
        # The match_id itself is not used for lookup; we re-normalize span.text.
        patterns_by_level: dict[str, list[Doc]] = {
            "province": [],
            "city": [],
            "district": [],
            "subdistrict": [],
        }

        for entry in self.gazetteer.entries:
            if entry.level not in allowed_levels:
                continue

            # Generate a handful of surface-form variants.
            variants = list(generate_surface_variants(entry.canonical_name))

            # City-only extra variants: allow matching "Surabaya" to "Kota Surabaya".
            if entry.level == "city":
                variants.extend(generate_city_surface_variants(entry.canonical_name))

            for variant in variants:
                patterns_by_level[entry.level].append(self.nlp.make_doc(variant))

        # Register patterns.
        for level, patterns in patterns_by_level.items():
            if patterns:
                self.matcher.add(level, patterns)


def _levels_up_to(max_level: str | None) -> set[str]:
    """Return allowed levels given a max granularity."""
    order = ["province", "city", "district", "subdistrict"]
    if max_level is None:
        return set(order)

    if max_level not in order:
        raise ValueError(f"Invalid max_level={max_level!r}. Expected one of: {order}")

    allowed: set[str] = set()
    for level in order:
        allowed.add(level)
        if level == max_level:
            break
    return allowed


def _pick_best_level(candidates: list[GazetteerEntry]) -> str:
    """Heuristic: choose the most specific (deepest) administrative level."""
    priority = {"province": 0, "city": 1, "district": 2, "subdistrict": 3}
    return max((c.level for c in candidates), key=lambda l: priority.get(l, -1))


def _candidate_dicts(
    candidates: list[GazetteerEntry],
    *,
    fuzzy_score: float | None = None,
) -> list[dict[str, Any]]:
    """Convert GazetteerEntry objects to JSON-friendly dictionaries."""
    out: list[dict[str, Any]] = []
    for c in candidates:
        d: dict[str, Any] = {
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
        if fuzzy_score is not None:
            d["fuzzy_score"] = fuzzy_score

        out.append(d)
    return out


def _infer_unique_code(mentions: list[LocationMention], code_field: str) -> str | None:
    """Infer a unique administrative code from extracted mentions.

    We use this to constrain fuzzy matching to a smaller search space.

    Example:
        If the text contains a single city (Kota Madiun), then when we fuzzy-match
        a district name we only search within that city.
    """
    codes: set[str] = set()
    for m in mentions:
        for c in m.candidates:
            v = c.get(code_field)
            if isinstance(v, str) and v.strip():
                codes.add(v.strip())

    if len(codes) == 1:
        return next(iter(codes))

    return None


def _overlaps_any(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    """Return True if span overlaps any span in others."""
    s1, e1 = span
    for s2, e2 in others:
        if s1 < e2 and s2 < e1:
            return True
    return False


def _infer_level_hint_from_left_context(text: str, start_char: int) -> str | None:
    """Infer a level hint from tokens immediately before a match.

    This helps disambiguate names that exist both as a district and a subdistrict,
    which is common in Indonesian administrative data.

    Example:
        "Kel. Cikole" -> hint "subdistrict"
        "Kec. Cikole" -> hint "district"

    This is intentionally conservative to avoid breaking general matching.

    Args:
        text: Full original text.
        start_char: Start position of the matched span.

    Returns:
        One of: province|city|district|subdistrict or None.
    """
    left = text[max(0, start_char - 25) : start_char].lower()

    # We only use hints for levels where the prefixes are unambiguous.
    if re.search(r"\b(kelurahan|kel\.?|desa)\s*$", left):
        return "subdistrict"

    if re.search(r"\b(kecamatan|kec\.?)\s*$", left):
        return "district"

    if re.search(r"\b(provinsi|propinsi)\s*$", left):
        return "province"

    # We intentionally do NOT add a city hint from "kota/kabupaten" here because
    # many official city names include those tokens and handling it correctly
    # requires span-aware parsing.
    return None


def _is_obvious_non_location(text: str, start_char: int, end_char: int) -> bool:
    """Heuristics to reduce common false positives.

    With a national gazetteer, many spans can be valid administrative names in
    *some* region, but in free-form text they may appear as part of other
    constructs (most commonly: street names).

    We keep this conservative and pattern-based (no hand-maintained stopwords).

    Args:
        text: Full original text.
        start_char: Match start.
        end_char: Match end.

    Returns:
        True if the match is almost certainly not an administrative location.
    """
    # Common address pattern: a token immediately after "Jl"/"Jalan" is very
    # often a street name, not an administrative unit.
    left_ctx = text[max(0, start_char - 20) : start_char].lower()
    if re.search(r"\b(jl\.?|jalan)\s*$", left_ctx):
        return True

    _ = text[start_char:end_char]  # kept for future heuristics
    return False


# Regex patterns to extract likely admin-name spans.
# We keep them conservative to avoid capturing long/irrelevant strings.
# Primary stop chars: comma, period, semicolon, newline.
# Secondary stop: if another admin keyword appears next (no punctuation).

_STOP_KEYWORDS_RE = re.compile(
    r"\b("
    r"kelurahan|kel\.?|desa|"
    r"kecamatan|kec\.?|"
    r"kota|kabupaten|kab\.?|"
    r"provinsi|propinsi|"
    r"rt|rw|"
    r"jalan|jl\.?|"
    r"no\.?|nomor"
    r")\b",
    re.IGNORECASE,
)

_PREFIX_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    # Subdistrict (Kelurahan/Desa)
    (
        "subdistrict",
        re.compile(r"\b(kelurahan|kel\.?|desa)\s+([^,.;\n]{2,60})", re.IGNORECASE),
        2,
    ),
    # District (Kecamatan)
    (
        "district",
        re.compile(r"\b(kecamatan|kec\.?)\s+([^,.;\n]{2,60})", re.IGNORECASE),
        2,
    ),
    # City/Regency (Kota/Kabupaten)
    (
        "city",
        re.compile(r"\b(kota|kabupaten|kab\.?)\s+([^,.;\n]{2,60})", re.IGNORECASE),
        0,
    ),
    # Province
    (
        "province",
        re.compile(r"\b(provinsi|propinsi)\s+([^,.;\n]{2,60})", re.IGNORECASE),
        2,
    ),
]


def _truncate_at_next_keyword(span_text: str) -> str:
    """Truncate a span when another keyword starts.

    This improves extraction when address text omits punctuation, e.g.:
        "Kelurahan Kejuron Kecamatan Taman Kota Madiun"

    Args:
        span_text: Extracted substring for a single prefixed span.

    Returns:
        Possibly truncated substring.
    """
    for m in _STOP_KEYWORDS_RE.finditer(span_text):
        # The first token for city spans is usually "kota"/"kabupaten" and is part of the name.
        # We only truncate when another keyword appears later.
        if m.start() == 0:
            continue
        return span_text[: m.start()]

    return span_text


def _extract_prefixed_spans(text: str) -> list[_PrefixedSpan]:
    """Extract spans like "Kelurahan X" / "Kecamatan Y" from the raw text."""
    spans: list[_PrefixedSpan] = []

    for level_hint, pattern, group_index in _PREFIX_PATTERNS:
        for m in pattern.finditer(text):
            if group_index == 0:
                start = m.start(0)
                raw = m.group(0)
            else:
                start = m.start(group_index)
                raw = m.group(group_index)

            raw = _truncate_at_next_keyword(raw)
            raw = raw.rstrip()
            if not raw:
                continue

            end = start + len(raw)

            spans.append(
                _PrefixedSpan(
                    level_hint=level_hint,
                    text=raw,
                    start_char=start,
                    end_char=end,
                )
            )

    # Sort by appearance.
    spans.sort(key=lambda x: (x.start_char, x.end_char))
    return spans
