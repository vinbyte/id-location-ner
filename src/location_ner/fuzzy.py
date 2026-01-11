"""Fuzzy matching utilities.

We use fuzzy matching to handle common user input issues:
- Typos ("Mediun" vs "Madiun")
- Minor spacing/punctuation differences

Implementation notes:
- This module depends on `rapidfuzz`, which is fast and lightweight.
- We keep the API small and testable.
"""

from __future__ import annotations

from typing import Iterable


def best_fuzzy_key(
    query: str,
    keys: Iterable[str],
    *,
    threshold: float,
) -> tuple[str | None, float | None]:
    """Return the best matching key and score.

    Args:
        query: Normalized query string.
        keys: Iterable of normalized candidate keys.
        threshold: Minimum score (0..100) to accept.

    Returns:
        (best_key, best_score). If nothing passes threshold, returns (None, None).
    """
    # Lazy import keeps rapidfuzz as a normal dependency but avoids import cost
    # if users never enable fuzzy mode.
    from rapidfuzz import fuzz, process  # type: ignore

    # `extractOne` returns: (match, score, index)
    result = process.extractOne(query, keys, scorer=fuzz.WRatio)
    if not result:
        return None, None

    match, score, _idx = result
    score = float(score)
    if score < threshold:
        return None, None

    return str(match), score
