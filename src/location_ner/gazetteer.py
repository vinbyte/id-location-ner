"""Gazetteer (location list) loader.

This module loads a Kemendagri-like CSV and builds lookup structures used by the
PhraseMatcher-based extractor.

Key design choice
-----------------
The raw CSV is usually *row-based* (one row per subdistrict/kelurahan), which
means province/city/district values repeat many times.

If we keep rows as-is, a match like "Jawa Timur" would return hundreds/thousands
of duplicate "candidates" (one per subdistrict row).

To avoid that, we *deduplicate* locations by their codes:
- province unique by `province_code`
- city unique by `city_code`
- district unique by `district_code`
- subdistrict unique by `subdistrict_code`

This yields clean, scalable output while still preserving the hierarchy.

Fuzzy matching support
----------------------
To handle typos, we also build per-level and per-parent indexes (e.g., districts
within a city). This allows fuzzy matching to be both:
- fast (smaller search space)
- safer (reduced false positives)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .fuzzy import best_fuzzy_key
from .normalization import generate_city_surface_variants, normalize_for_lookup


@dataclass(frozen=True)
class AdminLocation:
    """A unique administrative location node.

    Fields that are not applicable for the node's level are set to None.

    Examples:
        - province node: only province_* set
        - city node: province_* and city_* set
        - district node: province_* + city_* + district_* set
        - subdistrict node: all fields set (postal_code optional)
    """

    level: str  # province|city|district|subdistrict

    province_name: str
    province_code: str

    city_name: str | None
    city_code: str | None

    district_name: str | None
    district_code: str | None

    subdistrict_name: str | None
    subdistrict_code: str | None

    postal_code: str | None


@dataclass(frozen=True)
class GazetteerEntry:
    """A lookup entry for phrase matching (name -> AdminLocation)."""

    level: str  # province|city|district|subdistrict
    canonical_name: str
    normalized_name: str
    location: AdminLocation


class Gazetteer:
    """Holds all loaded locations and lookup maps."""

    def __init__(self, entries: list[GazetteerEntry]):
        self.entries = entries

        # normalized_name -> list[GazetteerEntry] across all levels
        lookup: dict[str, list[GazetteerEntry]] = {}
        for entry in entries:
            lookup.setdefault(entry.normalized_name, []).append(entry)
        self.lookup = lookup

        # Level-specific lookup maps.
        self.lookup_by_level: dict[str, dict[str, list[GazetteerEntry]]] = {}
        self.keys_by_level: dict[str, list[str]] = {}

        for level in {e.level for e in entries}:
            level_map: dict[str, list[GazetteerEntry]] = {}
            for e in entries:
                if e.level != level:
                    continue
                level_map.setdefault(e.normalized_name, []).append(e)
            self.lookup_by_level[level] = level_map
            self.keys_by_level[level] = list(level_map.keys())

        # Parent-constrained lookup maps (used for safer fuzzy matching).
        self.city_lookup_by_province: dict[str, dict[str, list[GazetteerEntry]]] = {}
        self.city_keys_by_province: dict[str, list[str]] = {}

        self.district_lookup_by_city: dict[str, dict[str, list[GazetteerEntry]]] = {}
        self.district_keys_by_city: dict[str, list[str]] = {}

        # Subdistricts can be constrained by district (strongest) or by city (fallback).
        self.subdistrict_lookup_by_district: dict[
            str, dict[str, list[GazetteerEntry]]
        ] = {}
        self.subdistrict_keys_by_district: dict[str, list[str]] = {}

        self.subdistrict_lookup_by_city: dict[str, dict[str, list[GazetteerEntry]]] = {}
        self.subdistrict_keys_by_city: dict[str, list[str]] = {}

        self._build_parent_indexes(entries)

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "Gazetteer":
        """Load a Kemendagri-like CSV into a Gazetteer.

        Args:
            csv_path: Path to the CSV.

        Returns:
            Gazetteer instance.
        """
        csv_path = Path(csv_path)

        provinces: dict[str, AdminLocation] = {}
        cities: dict[str, AdminLocation] = {}
        districts: dict[str, AdminLocation] = {}
        subdistricts: dict[str, AdminLocation] = {}

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            required_cols = {
                "province_name",
                "province_code",
                "city_name",
                "city_code",
                "district_name",
                "district_code",
                "subdistrict_name",
                "subdistrict_code",
            }
            missing = required_cols.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"CSV missing required columns: {sorted(missing)} (found={reader.fieldnames})"
                )

            for row in reader:
                province_name = row["province_name"].strip()
                province_code = row["province_code"].strip()
                city_name = row["city_name"].strip()
                city_code = row["city_code"].strip()
                district_name = row["district_name"].strip()
                district_code = row["district_code"].strip()
                subdistrict_name = row["subdistrict_name"].strip()
                subdistrict_code = row["subdistrict_code"].strip()
                postal_code = (row.get("postal_code") or "").strip() or None

                provinces.setdefault(
                    province_code,
                    AdminLocation(
                        level="province",
                        province_name=province_name,
                        province_code=province_code,
                        city_name=None,
                        city_code=None,
                        district_name=None,
                        district_code=None,
                        subdistrict_name=None,
                        subdistrict_code=None,
                        postal_code=None,
                    ),
                )

                cities.setdefault(
                    city_code,
                    AdminLocation(
                        level="city",
                        province_name=province_name,
                        province_code=province_code,
                        city_name=city_name,
                        city_code=city_code,
                        district_name=None,
                        district_code=None,
                        subdistrict_name=None,
                        subdistrict_code=None,
                        postal_code=None,
                    ),
                )

                districts.setdefault(
                    district_code,
                    AdminLocation(
                        level="district",
                        province_name=province_name,
                        province_code=province_code,
                        city_name=city_name,
                        city_code=city_code,
                        district_name=district_name,
                        district_code=district_code,
                        subdistrict_name=None,
                        subdistrict_code=None,
                        postal_code=None,
                    ),
                )

                subdistricts.setdefault(
                    subdistrict_code,
                    AdminLocation(
                        level="subdistrict",
                        province_name=province_name,
                        province_code=province_code,
                        city_name=city_name,
                        city_code=city_code,
                        district_name=district_name,
                        district_code=district_code,
                        subdistrict_name=subdistrict_name,
                        subdistrict_code=subdistrict_code,
                        postal_code=postal_code,
                    ),
                )

        entries: list[GazetteerEntry] = []
        entries.extend(_entries_for_locations(provinces.values()))
        entries.extend(_entries_for_locations(cities.values()))
        entries.extend(_entries_for_locations(districts.values()))
        entries.extend(_entries_for_locations(subdistricts.values()))

        return cls(entries=entries)

    def get_candidates(self, normalized_name: str) -> list[GazetteerEntry]:
        """Return all gazetteer candidates for a normalized key (all levels)."""
        return self.lookup.get(normalized_name, [])

    def get_candidates_at_level(
        self, level: str, normalized_name: str
    ) -> list[GazetteerEntry]:
        """Return candidates for a normalized key constrained to a level."""
        return self.lookup_by_level.get(level, {}).get(normalized_name, [])

    def fuzzy_match(
        self,
        level: str,
        query_normalized: str,
        *,
        threshold: float,
        province_code: str | None = None,
        city_code: str | None = None,
        district_code: str | None = None,
    ) -> tuple[list[GazetteerEntry], float | None, str | None]:
        """Fuzzy match a normalized query against the gazetteer.

        This returns the best matching key at the requested level, with optional
        parent constraints.

        Args:
            level: Target level (province|city|district|subdistrict).
            query_normalized: Normalized query string.
            threshold: Minimum score (0..100).
            province_code: Optional province constraint (useful for city matching).
            city_code: Optional city constraint (useful for district/subdistrict matching).
            district_code: Optional district constraint (useful for subdistrict matching).

        Returns:
            (candidates, score, matched_key)
        """
        lookup_map, keys = self._lookup_for_fuzzy(
            level,
            province_code=province_code,
            city_code=city_code,
            district_code=district_code,
        )

        if not lookup_map or not keys:
            return [], None, None

        matched_key, score = best_fuzzy_key(query_normalized, keys, threshold=threshold)
        if matched_key is None:
            return [], None, None

        return lookup_map.get(matched_key, []), score, matched_key

    def _lookup_for_fuzzy(
        self,
        level: str,
        *,
        province_code: str | None,
        city_code: str | None,
        district_code: str | None,
    ) -> tuple[dict[str, list[GazetteerEntry]], list[str]]:
        """Select the best lookup map for fuzzy matching given constraints."""
        if level == "city" and province_code:
            m = self.city_lookup_by_province.get(province_code, {})
            return m, self.city_keys_by_province.get(province_code, [])

        if level == "district" and city_code:
            m = self.district_lookup_by_city.get(city_code, {})
            return m, self.district_keys_by_city.get(city_code, [])

        if level == "subdistrict" and district_code:
            m = self.subdistrict_lookup_by_district.get(district_code, {})
            return m, self.subdistrict_keys_by_district.get(district_code, [])

        # If district is unknown but city is known, this is still a useful constraint.
        if level == "subdistrict" and city_code:
            m = self.subdistrict_lookup_by_city.get(city_code, {})
            return m, self.subdistrict_keys_by_city.get(city_code, [])

        # Fallback: level-wide search.
        m = self.lookup_by_level.get(level, {})
        return m, self.keys_by_level.get(level, [])

    def _build_parent_indexes(self, entries: list[GazetteerEntry]) -> None:
        """Build parent-constrained lookup indexes."""
        # Cities grouped by province.
        for e in entries:
            if e.level != "city":
                continue
            province_code = e.location.province_code
            self.city_lookup_by_province.setdefault(province_code, {}).setdefault(
                e.normalized_name, []
            ).append(e)

        for province_code, m in self.city_lookup_by_province.items():
            self.city_keys_by_province[province_code] = list(m.keys())

        # Districts grouped by city.
        for e in entries:
            if e.level != "district":
                continue
            if not e.location.city_code:
                continue
            city_code = e.location.city_code
            self.district_lookup_by_city.setdefault(city_code, {}).setdefault(
                e.normalized_name, []
            ).append(e)

        for city_code, m in self.district_lookup_by_city.items():
            self.district_keys_by_city[city_code] = list(m.keys())

        # Subdistricts grouped by district (best constraint) and by city (fallback constraint).
        for e in entries:
            if e.level != "subdistrict":
                continue

            if e.location.district_code:
                district_code = e.location.district_code
                self.subdistrict_lookup_by_district.setdefault(
                    district_code, {}
                ).setdefault(e.normalized_name, []).append(e)

            if e.location.city_code:
                city_code = e.location.city_code
                self.subdistrict_lookup_by_city.setdefault(city_code, {}).setdefault(
                    e.normalized_name, []
                ).append(e)

        for district_code, m in self.subdistrict_lookup_by_district.items():
            self.subdistrict_keys_by_district[district_code] = list(m.keys())

        for city_code, m in self.subdistrict_lookup_by_city.items():
            self.subdistrict_keys_by_city[city_code] = list(m.keys())


def _entries_for_locations(locations: Iterable[AdminLocation]) -> list[GazetteerEntry]:
    """Build GazetteerEntry objects for a collection of unique locations.

    For city/regency names, we also add alias keys (e.g., "Surabaya" -> "Kota Surabaya")
    so that exact matching can resolve candidates even when the prefix is omitted.
    """
    entries: list[GazetteerEntry] = []

    for loc in locations:
        name = _name_for_level(loc)
        normalized = normalize_for_lookup(name)
        if not normalized:
            continue

        # Always add the canonical key.
        entries.append(
            GazetteerEntry(
                level=loc.level,
                canonical_name=name,
                normalized_name=normalized,
                location=loc,
            )
        )

        # City-only aliases: allow lookups by stripped name (e.g., "Surabaya").
        if loc.level == "city":
            seen_keys: set[str] = {normalized}
            for alias in generate_city_surface_variants(name):
                alias_norm = normalize_for_lookup(alias)
                if not alias_norm or alias_norm in seen_keys:
                    continue
                seen_keys.add(alias_norm)
                entries.append(
                    GazetteerEntry(
                        level=loc.level,
                        canonical_name=name,
                        normalized_name=alias_norm,
                        location=loc,
                    )
                )

    return entries


def _name_for_level(loc: AdminLocation) -> str:
    """Return the display name for the node based on its level."""
    if loc.level == "province":
        return loc.province_name
    if loc.level == "city":
        return loc.city_name or ""
    if loc.level == "district":
        return loc.district_name or ""
    if loc.level == "subdistrict":
        return loc.subdistrict_name or ""
    raise ValueError(f"Unknown location level: {loc.level!r}")
