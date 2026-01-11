"""Text normalization utilities.

We use lightweight normalization to make matching more robust:
- Case-insensitive matching
- Hyphen and punctuation tolerance (e.g., "Oro-oro" vs "Oro oro")

This is intentionally simple and fast. For production usage, you may extend it
with a curated synonym/alias list.
"""

from __future__ import annotations

import re

# Match any sequence of whitespace.
_WHITESPACE_RE = re.compile(r"\s+")

# Keep letters/digits/spaces, drop other punctuation/symbols.
# NOTE: This is a pragmatic choice to handle inputs like "Oro-oro".
_NON_ALNUM_SPACE_RE = re.compile(r"[^0-9A-Za-z\s]+", flags=re.UNICODE)

# Used to truncate spans like "kelurahan X kecamatan Y" -> "X".
# This is shared by fuzzy matching and NER-to-gazetteer linking.
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
    flags=re.IGNORECASE,
)


def normalize_for_lookup(text: str) -> str:
    """Normalize a string for dictionary lookup.

    The goal is to reduce common surface-form differences while keeping enough
    signal for accurate matching.

    Examples:
        "Oro-oro Ombo"  -> "oro oro ombo"
        "  Kota   Madiun" -> "kota madiun"

    Args:
        text: Raw input text.

    Returns:
        A normalized string.
    """
    text = text.strip().lower()

    # Treat hyphens as spaces, then remove remaining punctuation.
    # This makes "oro-oro" and "oro oro" equivalent.
    text = text.replace("-", " ")
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)

    # Collapse multiple spaces.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def strip_admin_prefix(text: str) -> tuple[str, str | None]:
    """Strip common administrative prefixes from an entity span.

    Pretrained NER models often output spans like:
        - "Kelurahan Kejuron"
        - "Kecamatan Taman"

    But your gazetteer typically stores just the canonical name:
        - "Kejuron"
        - "Taman"

    This helper removes a small, safe set of Indonesian admin prefixes and returns:
    - the stripped (normalized) text
    - an optional `level_hint` you can use to disambiguate candidates

    IMPORTANT:
    We intentionally do NOT strip "kota" / "kabupaten" by default, because many
    official city names in the gazetteer include those tokens (e.g., "Kota Madiun").

    Args:
        text: Raw entity text (any casing/punctuation).

    Returns:
        (normalized_stripped_text, level_hint)
        where level_hint is one of: province|district|subdistrict or None.
    """
    normalized = normalize_for_lookup(text)
    if not normalized:
        return normalized, None

    tokens = normalized.split()
    if not tokens:
        return normalized, None

    # Map a prefix token to the administrative level it implies.
    # Keep this list small and conservative to avoid false stripping.
    prefix_to_level = {
        # Province
        "provinsi": "province",
        "propinsi": "province",
        # District (kecamatan)
        "kecamatan": "district",
        "kec": "district",
        # Subdistrict (kelurahan/desa)
        "kelurahan": "subdistrict",
        "kel": "subdistrict",
        "desa": "subdistrict",
    }

    first = tokens[0]
    level_hint = prefix_to_level.get(first)
    if level_hint is None:
        return normalized, None

    # Drop the prefix token.
    stripped_tokens = tokens[1:]
    stripped = " ".join(stripped_tokens).strip()

    # If the NER model produced a longer span (no punctuation), truncate at the next keyword.
    # Example: "kelurahan pangongngan kecamatan manguharjo" -> "pangongngan".
    m = _STOP_KEYWORDS_RE.search(stripped)
    if m:
        stripped = stripped[: m.start()].rstrip()

    return stripped, level_hint


def generate_city_surface_variants(city_name: str) -> list[str]:
    """Generate extra variants specifically for Indonesian city/regency names.

    Many official names in Kemendagri-style gazetteers include administrative
    prefixes such as "Kota" or "Kabupaten":
        - "Kota Surabaya"
        - "Kabupaten Sidoarjo"

    In real-world addresses, people often omit the prefix and write only:
        - "Surabaya"
        - "Sidoarjo"

    This helper generates safe variants by stripping those prefixes.

    IMPORTANT:
    Stripping prefixes can introduce ambiguity (e.g., "Madiun" could be a city
    or a district name). We only use these variants for *city-level* patterns,
    and downstream resolution still uses other mentions (kecamatan/kelurahan)
    to pick a consistent final path.

    Args:
        city_name: Canonical city/regency name from the gazetteer.

    Returns:
        A list of variant strings (may be empty).
    """
    city_name = city_name.strip()
    if not city_name:
        return []

    lowered = city_name.lower().strip()

    prefixes = [
        "kota ",
        "kabupaten ",
        "kab. ",
        "kab ",
    ]

    stripped: list[str] = []
    for p in prefixes:
        if lowered.startswith(p):
            # Keep original casing after stripping by slicing the original string.
            stripped_name = city_name[len(p) :].strip()
            if stripped_name:
                stripped.append(stripped_name)

    # Deduplicate while preserving a stable order.
    seen: set[str] = set()
    out: list[str] = []
    for v in stripped:
        v2 = _WHITESPACE_RE.sub(" ", v).strip()
        if v2 and v2 not in seen:
            seen.add(v2)
            out.append(v2)

    return out


def generate_surface_variants(name: str) -> list[str]:
    """Generate a small set of common surface-form variants for a gazetteer name.

    This is a *cheap* way to improve matching without ML.

    Args:
        name: Canonical location name from the gazetteer.

    Returns:
        A list of variant strings (including the original).
    """
    name = name.strip()
    if not name:
        return []

    variants: set[str] = {name}

    # Common punctuation variants.
    variants.add(name.replace("-", " "))
    variants.add(name.replace("'", ""))
    variants.add(name.replace(".", ""))

    # If the name contains hyphens, also try removing them entirely.
    if "-" in name:
        variants.add(name.replace("-", ""))

    # Normalize whitespace in each variant.
    cleaned: set[str] = set()
    for v in variants:
        v2 = _WHITESPACE_RE.sub(" ", v).strip()
        if v2:
            cleaned.add(v2)

    return sorted(cleaned)
