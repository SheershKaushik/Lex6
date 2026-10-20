"""Country-name detection in queries and chunk text (pycountry names plus common aliases)."""

from __future__ import annotations

import re
from functools import lru_cache

import pycountry

_ALIASES: dict[str, tuple[str, ...]] = {
    "US": ("USA", "U.S.", "U.S.A.", "US", "United States of America"),
    "GB": ("UK", "U.K.", "Britain", "Great Britain"),
    "RU": ("Russia",),
    "KR": ("Republic of Korea",),
    "KP": ("North Korea", "DPRK"),
    "TR": ("Turkey", "Turkiye"),
    "CZ": ("Czech Republic",),
    "CI": ("Ivory Coast", "Cote d'Ivoire"),
    "CD": ("DR Congo", "DRC", "Democratic Republic of the Congo"),
    "CG": ("Republic of the Congo",),
    "SZ": ("Swaziland",),
    "MK": ("Macedonia",),
    "CV": ("Cape Verde",),
    "MM": ("Burma",),
    "PS": ("Palestine",),
    "VA": ("Vatican",),
    "BN": ("Brunei",),
    "MD": ("Moldova",),
    "TZ": ("Tanzania",),
    "VE": ("Venezuela",),
    "FM": ("Micronesia",),
}
# Names that are also common lowercase words: only match when capitalised.
_CASE_SENSITIVE = {"Chad", "Jordan", "Georgia", "Turkey", "China", "Japan", "Jersey", "Guernsey", "Panama", "Niger"}


@lru_cache
def country_variants() -> dict[str, tuple[str, ...]]:
    """Canonical country name -> all names it may appear under (inverted ``"Korea, Republic of"`` forms dropped)."""
    out: dict[str, tuple[str, ...]] = {}
    for c in pycountry.countries:
        canonical = getattr(c, "common_name", None) or c.name
        names = {getattr(c, a) for a in ("name", "common_name", "official_name") if getattr(c, a, None)}
        names |= set(_ALIASES.get(c.alpha_2, ()))
        out[canonical] = tuple(sorted(n for n in names if ", " not in n))
    return out


def _pattern(name: str) -> re.Pattern[str]:
    case_sensitive = name in _CASE_SENSITIVE or name.replace(".", "").isupper()
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", 0 if case_sensitive else re.IGNORECASE)


@lru_cache
def _patterns() -> list[tuple[str, re.Pattern[str]]]:
    pats = [(name, canonical) for canonical, names in country_variants().items() for name in names]
    pats.sort(key=lambda p: -len(p[0]))
    return [(canonical, _pattern(name)) for name, canonical in pats]


@lru_cache
def _patterns_for(canonical: str) -> tuple[re.Pattern[str], ...]:
    return tuple(_pattern(n) for n in country_variants().get(canonical, ()))


@lru_cache
def _by_lower_name() -> dict[str, str]:
    return {n.lower(): canonical for canonical, names in country_variants().items() for n in names}


def canonical_country(name: str | None) -> str | None:
    return _by_lower_name().get(name.lower()) if name else None


def extract_countries(text: str) -> list[str]:
    """Canonical names of the countries mentioned in ``text``, longest names first so "Papua New Guinea" isn't
    also read as "Guinea"."""
    found: list[str] = []
    for canonical, pat in _patterns():
        text, n = pat.subn(lambda m: " " * len(m.group(0)), text)
        if n and canonical not in found:
            found.append(canonical)
    return found


def mentions_country(text: str, canonical: str) -> bool:
    if not any(p.search(text) for p in _patterns_for(canonical)):
        return False
    return canonical in extract_countries(text)
