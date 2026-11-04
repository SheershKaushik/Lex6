"""Country comparison for the Compare view, read fresh from the country workbook (``data/enriched_data.xlsx``).

The workbook is re-read on every request, so edits by the data team show up without restarting the server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# LLM prompt text and raw IPCC-engagement working columns are not comparison items (nor is Party, the row key).
EXCLUDED_COLUMNS = {
    "IPCC Engagement Query",
    "IPCC Engagement Level",
    "IPCC",
    "Lex6 Engagement Level",
    "Lex6",
    "G LLM Engagement Level",
    "General LLM",
    "E LLM Engagement Level",
    "Expert LLM",
}
# Assessment factors shown right after the country profile (the order the user listed them); the rest follow in
# workbook order.
HEADLINE_FACTORS = [
    "Best Available Science",
    "Clarity of the Headline Target",
    "Reliance on Article 6 for Achieving the NDC",
    "Baseline Integrity",
    "Sector Coverage",
    "Adaptation Coverage",
    "Legal Status",
    "Stakeholder Consultation",
    "Monitoring Reporting and Verification",
    "Net-Zero Target Quality",
    "Strength of Commitment Language",
    "Level of IPCC Engagement",
]
PROFILE_END = "additional_document_urls"  # last country-profile column; assessment factors come after it
COUNTRY_COLUMN = "Party"


class WorkbookError(RuntimeError):
    pass


@dataclass
class Column:
    index: int
    header: str
    label: str
    group: str


def label_for(header: str) -> str:
    return re.sub(r"\s+", " ", header).strip()


def comparison_columns(headers: list[Any]) -> list[Column]:
    """Comparison items in display order: country profile, headline factors, then the remaining factors."""
    profile: list[Column] = []
    factors: list[Column] = []
    in_profile = True
    for i, h in enumerate(headers):
        if h is None or not str(h).strip():
            continue
        label = label_for(str(h))
        if label != COUNTRY_COLUMN and not label.lower().startswith("query") and label not in EXCLUDED_COLUMNS:
            (profile if in_profile else factors).append(
                Column(i, str(h), label, "Country profile" if in_profile else "NDC assessment")
            )
        if label == PROFILE_END:
            in_profile = False
    rank = {name.lower(): n for n, name in enumerate(HEADLINE_FACTORS)}
    factors.sort(key=lambda c: (rank.get(c.label.lower(), len(rank)), c.index))
    return profile + factors


def format_value(value: Any, number_format: str = "General") -> str | None:
    """Render a cell the way Excel displays it (percentages, thousands separators, dates)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return "Yes" if value else "No"
    fmt = (number_format or "General").split(";")[0]
    if isinstance(value, (int, float)):
        if "%" in fmt:
            decimals = len(fmt.split(".")[1].split("%")[0]) if "." in fmt else 0
            return f"{value * 100:.{decimals}f}%"
        if "#,##0" in fmt:
            decimals = len(re.match(r"0*", fmt.split(".")[1]).group()) if "." in fmt else 0
            return f"{value:,.{decimals}f}"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value).replace("\xa0", " ").strip()


class CountryWorkbook:
    """One fresh read of the workbook: comparison columns and a row per country."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise WorkbookError(f"country workbook not found: {self.path}")
        import openpyxl

        try:
            wb = openpyxl.load_workbook(self.path, read_only=True, data_only=True)
        except Exception as e:
            raise WorkbookError(f"could not read {self.path.name}: {e}") from e
        try:
            rows = list(wb.active.iter_rows())
        finally:
            wb.close()
        if not rows:
            raise WorkbookError(f"{self.path.name} is empty")
        headers = [c.value for c in rows[0]]
        labels = [label_for(str(h)) if h is not None else "" for h in headers]
        if COUNTRY_COLUMN not in labels:
            raise WorkbookError(f"{self.path.name} has no '{COUNTRY_COLUMN}' column")
        self.columns = comparison_columns(headers)
        self.modified = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc).isoformat()
        key = labels.index(COUNTRY_COLUMN)
        self.rows: dict[str, list[Any]] = {}
        for row in rows[1:]:
            name = row[key].value if key < len(row) else None
            if name is not None and str(name).strip():
                self.rows[str(name).strip()] = row

    def countries(self) -> list[str]:
        return sorted(self.rows, key=str.casefold)

    def compare(self, countries: list[str]) -> dict[str, Any]:
        missing = [c for c in countries if c not in self.rows]
        if missing:
            raise KeyError(", ".join(missing))
        items = []
        for col in self.columns:
            values = []
            for c in countries:
                row = self.rows[c]
                cell = row[col.index] if col.index < len(row) else None
                values.append(format_value(getattr(cell, "value", None), getattr(cell, "number_format", "General")))
            items.append({"label": col.label, "group": col.group, "values": values})
        return {"countries": countries, "items": items, "source": self.path.name, "modified": self.modified}
