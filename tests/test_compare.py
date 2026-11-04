from datetime import datetime

import openpyxl
import pytest

from lex6.compare import CountryWorkbook, WorkbookError, comparison_columns, format_value

HEADERS = [
    "ISO2",
    "Party",
    "Population 2026",
    "Urban Pop %",
    "JCM\nProjects",
    "submission_date",
    "additional_document_urls",
    "IPCC Engagement Query",
    "Expert LLM",
    "Legal Status",
    "Query for Legal Status",
    "Leakage",
    "Query for Leakage",
    "Clarity of the Headline Target",
    "Query for Clarity of the Headline Target",
]


def write_workbook(path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(HEADERS)
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows(min_row=2):
        row[2].number_format = "#,##0"
        row[3].number_format = "0.00%"
    wb.save(path)


ROWS = [
    [
        "ZM",
        "Zambia",
        21913874,
        0.462,
        0,
        datetime(2025, 9, 22),
        None,
        "q",
        "x",
        "partial",
        "p",
        "minimal",
        "p",
        "substantial",
        "p",
    ],
    [
        "AF",
        "Afghanistan",
        45047068,
        0.262,
        3,
        datetime(2016, 11, 23),
        "https://a https://b",
        "q",
        "x",
        "partial",
        "p",
        None,
        "p",
        "minimal",
        "p",
    ],
]


def test_columns_drop_queries_and_working_columns_and_order_headline_factors():
    cols = comparison_columns(HEADERS)
    labels = [c.label for c in cols]
    assert labels == [
        "ISO2",
        "Population 2026",
        "Urban Pop %",
        "JCM Projects",
        "submission_date",
        "additional_document_urls",
        "Clarity of the Headline Target",
        "Legal Status",
        "Leakage",
    ]
    assert {c.group for c in cols[:6]} == {"Country profile"} and {c.group for c in cols[6:]} == {"NDC assessment"}


@pytest.mark.parametrize(
    ("value", "fmt", "expected"),
    [
        (None, "General", None),
        ("  ", "General", None),
        (45047068, "#,##0", "45,047,068"),
        (0.0274, "0.00%", "2.74%"),
        (1, "0.00%", "100.00%"),
        (0, "#,##0;\\-#,##0;\\-", "0"),
        (4.56, "General", "4.56"),
        (datetime(2016, 11, 23), "yyyy-mm-dd", "2016-11-23"),
        ("<\xa00.01%", "General", "< 0.01%"),
    ],
)
def test_format_value_matches_excel_display(value, fmt, expected):
    assert format_value(value, fmt) == expected


def test_workbook_compare_and_fresh_reads(tmp_path):
    path = tmp_path / "enriched_data.xlsx"
    write_workbook(path, ROWS)
    wb = CountryWorkbook(path)
    assert wb.countries() == ["Afghanistan", "Zambia"]
    result = wb.compare(["Zambia", "Afghanistan"])
    values = {it["label"]: it["values"] for it in result["items"]}
    assert values["Population 2026"] == ["21,913,874", "45,047,068"]
    assert values["Urban Pop %"] == ["46.20%", "26.20%"]
    assert values["Leakage"] == ["minimal", None]
    assert result["source"] == "enriched_data.xlsx"
    with pytest.raises(KeyError):
        wb.compare(["Zambia", "Atlantis"])

    ROWS[0][9] = "substantial"
    write_workbook(path, ROWS)
    values = {it["label"]: it["values"] for it in CountryWorkbook(path).compare(["Zambia", "Afghanistan"])["items"]}
    assert values["Legal Status"] == ["substantial", "partial"]


def test_missing_workbook(tmp_path):
    with pytest.raises(WorkbookError, match="not found"):
        CountryWorkbook(tmp_path / "nope.xlsx")


def test_repo_workbook_loads():
    from lex6.config import PROJECT_ROOT

    wb = CountryWorkbook(PROJECT_ROOT / "data" / "enriched_data.xlsx")
    assert len(wb.countries()) > 150 and "Zambia" in wb.countries()
    assert not any(c.label.startswith("Query") for c in wb.columns)
    assert len(wb.columns) == 91
