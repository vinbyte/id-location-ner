"""Resolve a single best administrative location from multiple mentions.

Problem
-------
The extractor may return multiple mentions, and each mention may be ambiguous.
Example:
- "Sumberejo" might exist in multiple districts/cities
- "Pakal" might exist in multiple cities

Downstream APIs often need *one* final administrative path:
(province, city, district, subdistrict) + their codes.

Approach
--------
We treat every candidate as a *potential administrative path*.
Then we score each path using evidence from the extracted mentions.

Key ideas:
- Consistency: a candidate is compatible if it does not contradict the path.
- Evidence weighting: highly ambiguous mentions (many candidates) contribute less.
- Specificity: deeper levels are preferred, but only when supported.

This is a heuristic (not a ML disambiguation model), but it works well for
address-like text where multiple levels co-occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_CODE_FIELDS = [
    "province_code",
    "city_code",
    "district_code",
    "subdistrict_code",
]


@dataclass(frozen=True)
class ResolvedLocation:
    """Final resolved administrative path."""

    province_name: str | None
    province_code: str | None
    city_name: str | None
    city_code: str | None
    district_name: str | None
    district_code: str | None
    subdistrict_name: str | None
    subdistrict_code: str | None

    @classmethod
    def from_candidate_dict(cls, c: dict[str, Any]) -> "ResolvedLocation":
        return cls(
            province_name=_as_str_or_none(c.get("province_name")),
            province_code=_as_str_or_none(c.get("province_code")),
            city_name=_as_str_or_none(c.get("city_name")),
            city_code=_as_str_or_none(c.get("city_code")),
            district_name=_as_str_or_none(c.get("district_name")),
            district_code=_as_str_or_none(c.get("district_code")),
            subdistrict_name=_as_str_or_none(c.get("subdistrict_name")),
            subdistrict_code=_as_str_or_none(c.get("subdistrict_code")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "province_name": self.province_name,
            "province_code": self.province_code,
            "city_name": self.city_name,
            "city_code": self.city_code,
            "district_name": self.district_name,
            "district_code": self.district_code,
            "subdistrict_name": self.subdistrict_name,
            "subdistrict_code": self.subdistrict_code,
        }


def resolve_best_location(mentions: list[dict[str, Any]]) -> ResolvedLocation | None:
    """Pick the single best administrative path from extracted mentions.

    Args:
        mentions: List of mentions as dictionaries.
            This is compatible with the CLI output: `gazetteer_mentions`.
            Each mention must contain `candidates` (list of candidate dicts).

    Returns:
        A ResolvedLocation, or None if there are no candidates at all.
    """
    all_candidates: list[dict[str, Any]] = []
    for m in mentions:
        for c in m.get("candidates", []) or []:
            if isinstance(c, dict):
                all_candidates.append(c)

    if not all_candidates:
        return None

    # Use each candidate as a potential "path" to explain all mentions.
    best: dict[str, Any] | None = None
    best_score: float = float("-inf")

    for path in all_candidates:
        score = _score_path(path, mentions)
        if score > best_score:
            best = path
            best_score = score
        elif score == best_score and best is not None:
            # Tie-break: prefer deeper (more specific) paths.
            if _specificity_rank(path) > _specificity_rank(best):
                best = path

    if best is None:
        return None

    resolved = ResolvedLocation.from_candidate_dict(best)
    return _prune_unreliable_levels(resolved, mentions)


def _score_path(path: dict[str, Any], mentions: list[dict[str, Any]]) -> float:
    """Score how well a path explains the mentions.

    This scoring method is designed for national gazetteers where some surface
    forms are extremely ambiguous (e.g., "Sumberejo").

    We downweight ambiguous mentions by a factor derived from the number of
    candidates for that mention.
    """
    score = 0.0
    supported_mentions = 0

    for m in mentions:
        candidates = [c for c in (m.get("candidates") or []) if isinstance(c, dict)]
        if not candidates:
            continue

        n = max(1, len(candidates))

        # Ambiguity penalty: a mention with many candidates should contribute less.
        # sqrt keeps the penalty mild enough for real-world data.
        mention_weight = 1.0 / (n**0.5)

        match_method = _as_str_or_none(m.get("match_method"))
        method_bonus = {"exact": 0.3, "fuzzy": 0.1}.get(match_method or "", 0.0)

        best_for_mention = 0.0
        for c in candidates:
            if not _compatible(path, c):
                continue

            level = _as_str_or_none(c.get("level"))
            level_weight = {
                "province": 1.0,
                "city": 2.0,
                "district": 3.0,
                "subdistrict": 4.0,
            }.get(level or "", 1.0)

            fuzzy_score = c.get("fuzzy_score")
            fuzzy_bonus = 0.0
            if isinstance(fuzzy_score, (int, float)):
                # Normalize to a small bonus when fuzzy score is high.
                fuzzy_bonus = max(0.0, min(0.25, (float(fuzzy_score) - 85.0) / 100.0))

            evidence = mention_weight * (level_weight + method_bonus + fuzzy_bonus)
            best_for_mention = max(best_for_mention, evidence)

        if best_for_mention > 0.0:
            supported_mentions += 1

        score += best_for_mention

    # Prefer paths supported by multiple mentions. This helps avoid "single-word"
    # traps where a common token happens to be a valid subdistrict somewhere.
    score += 0.75 * max(0, supported_mentions - 1)

    # Small preference for specificity (does not override evidence).
    score += 0.1 * _specificity_rank(path)
    return score


def _specificity_rank(c: dict[str, Any]) -> int:
    """Return an integer rank where higher means more specific."""
    if _as_str_or_none(c.get("subdistrict_code")):
        return 3
    if _as_str_or_none(c.get("district_code")):
        return 2
    if _as_str_or_none(c.get("city_code")):
        return 1
    if _as_str_or_none(c.get("province_code")):
        return 0
    return -1


def _compatible(path: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Return True if candidate does not contradict the path.

    A candidate is compatible when for every code field it specifies, the path
    has the same value.
    """
    for field in _CODE_FIELDS:
        v_candidate = _as_str_or_none(candidate.get(field))
        if v_candidate is None:
            continue

        v_path = _as_str_or_none(path.get(field))
        if v_path is None:
            # The path is too generic to explain this candidate.
            return False

        if v_candidate != v_path:
            return False

    return True


def _prune_unreliable_levels(
    resolved: ResolvedLocation,
    mentions: list[dict[str, Any]],
) -> ResolvedLocation:
    """Drop deep levels that are not reliably supported.

    User requirement: if subdistrict cannot be resolved confidently, leave it
    empty (null) instead of forcing a best-guess.

    Rule:
    - Keep `subdistrict_*` only if there is at least one mention whose candidates
      contain the same `subdistrict_code` AND that mention came from a strong
      signal (fuzzy match or explicit level hint).

    This prevents a common failure mode where a highly ambiguous exact match
    (e.g., a common village name) causes an arbitrary subdistrict to be chosen.
    """

    # If there is no subdistrict in the resolved path, nothing to prune.
    if resolved.subdistrict_code is None:
        return resolved

    supported = False
    for m in mentions:
        candidates = [c for c in (m.get("candidates") or []) if isinstance(c, dict)]
        if not candidates:
            continue

        # Strong signals always count.
        strong_signal = (m.get("level_hint") is not None) or (
            m.get("match_method") == "fuzzy"
        )

        if strong_signal:
            for c in candidates:
                if (
                    _as_str_or_none(c.get("subdistrict_code"))
                    == resolved.subdistrict_code
                ):
                    supported = True
                    break

            if supported:
                break

        # Otherwise, allow support when the mention becomes unambiguous AFTER we
        # apply the already-resolved higher-level codes.
        narrowed = _narrow_by_resolved_prefix(resolved, candidates)
        if (
            len(narrowed) == 1
            and _as_str_or_none(narrowed[0].get("subdistrict_code"))
            == resolved.subdistrict_code
        ):
            supported = True
            break

    if supported:
        return resolved

    # Drop subdistrict fields.
    return ResolvedLocation(
        province_name=resolved.province_name,
        province_code=resolved.province_code,
        city_name=resolved.city_name,
        city_code=resolved.city_code,
        district_name=resolved.district_name,
        district_code=resolved.district_code,
        subdistrict_name=None,
        subdistrict_code=None,
    )


def _narrow_by_resolved_prefix(
    resolved: ResolvedLocation,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Narrow candidates using already resolved higher-level codes.

    This makes it safe to accept a subdistrict match even when the mention itself
    has no explicit prefix (e.g., just "Cisarua"), as long as only one candidate
    remains after applying the resolved province/city/district.
    """
    out: list[dict[str, Any]] = []

    for c in candidates:
        province_code = _as_str_or_none(c.get("province_code"))
        city_code = _as_str_or_none(c.get("city_code"))
        district_code = _as_str_or_none(c.get("district_code"))

        if (
            resolved.province_code is not None
            and province_code is not None
            and province_code != resolved.province_code
        ):
            continue
        if (
            resolved.city_code is not None
            and city_code is not None
            and city_code != resolved.city_code
        ):
            continue
        if (
            resolved.district_code is not None
            and district_code is not None
            and district_code != resolved.district_code
        ):
            continue

        out.append(c)

    return out


_LEVEL_PRIORITY = {"province": 0, "city": 1, "district": 2, "subdistrict": 3}


def prune_mentions_to_resolved(
    mentions: list[dict[str, Any]],
    resolved: ResolvedLocation | None,
) -> list[dict[str, Any]]:
    """Keep only mentions consistent with the chosen best path.

    Motivation:
    - With a national gazetteer, many common words are valid locations somewhere.
    - For typical address-like text, downstream users usually want the *single*
      best administrative path and the mentions that support it.

    This function:
    - Filters each mention's candidates to those compatible with `resolved`.
    - Drops mentions that have no remaining candidates.
    - Recomputes mention `level` from the remaining candidates.

    Args:
        mentions: Extracted mentions in CLI-compatible dict format.
        resolved: Output of `resolve_best_location`.

    Returns:
        A pruned list of mentions.
    """
    if resolved is None:
        return mentions

    out: list[dict[str, Any]] = []

    for m in mentions:
        candidates = [c for c in (m.get("candidates") or []) if isinstance(c, dict)]
        if not candidates:
            continue

        narrowed = _narrow_candidate_dicts_to_resolved(resolved, candidates)
        if not narrowed:
            continue

        new_m = dict(m)
        new_m["candidates"] = narrowed

        best_level = _pick_best_level_from_candidate_dicts(narrowed)
        if best_level is not None:
            new_m["level"] = best_level

        out.append(new_m)

    return out


def _narrow_candidate_dicts_to_resolved(
    resolved: ResolvedLocation,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter candidates to those that match the resolved codes.

    We only apply constraints that are present on the resolved path.
    """
    out: list[dict[str, Any]] = []
    for c in candidates:
        province_code = _as_str_or_none(c.get("province_code"))
        city_code = _as_str_or_none(c.get("city_code"))
        district_code = _as_str_or_none(c.get("district_code"))
        subdistrict_code = _as_str_or_none(c.get("subdistrict_code"))

        if (
            resolved.province_code is not None
            and province_code is not None
            and province_code != resolved.province_code
        ):
            continue
        if (
            resolved.city_code is not None
            and city_code is not None
            and city_code != resolved.city_code
        ):
            continue
        if (
            resolved.district_code is not None
            and district_code is not None
            and district_code != resolved.district_code
        ):
            continue
        if (
            resolved.subdistrict_code is not None
            and subdistrict_code is not None
            and subdistrict_code != resolved.subdistrict_code
        ):
            continue

        out.append(c)

    return out


def _pick_best_level_from_candidate_dicts(
    candidates: list[dict[str, Any]],
) -> str | None:
    best: str | None = None
    best_rank = -1

    for c in candidates:
        level = _as_str_or_none(c.get("level"))
        if level is None:
            continue
        rank = _LEVEL_PRIORITY.get(level, -1)
        if rank > best_rank:
            best = level
            best_rank = rank

    return best


def _as_str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return None
