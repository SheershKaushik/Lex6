"""``lex6`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _cmd_ingest(args: argparse.Namespace) -> int:
    from lex6.services import build_ingestion_pipeline

    reports = build_ingestion_pipeline().ingest(args.paths)
    for r in reports:
        line = f"[{r.status:8s}] {r.filename}"
        if r.status == "ingested":
            line += f"  lang={r.language} translated_pages={r.translated_pages} pages={r.pages} chunks={r.chunks}"
        if r.errors:
            line += "  errors: " + "; ".join(r.errors)
        print(line)
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    return 0 if all(r.status == "ingested" for r in reports) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lex6", description="Agentic RAG over climate-law documents")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Validate, translate, chunk, embed and store files/directories")
    p.add_argument("paths", nargs="+")
    p.add_argument("--json", action="store_true", help="Also print the reports as JSON")
    p.set_defaults(func=_cmd_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
