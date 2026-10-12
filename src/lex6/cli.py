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


def _cmd_ask(args: argparse.Namespace) -> int:
    from lex6.rag import rag_advanced
    from lex6.services import build_llm, build_retriever

    retriever = build_retriever(args.retriever)
    min_score = args.min_score if args.min_score is not None else (0.2 if args.retriever == "naive" else -10.0)
    result = rag_advanced(args.question, retriever, build_llm(), top_k=args.top_k, min_score=min_score)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        page = f" p.{s['page']}" if s["page"] is not None else ""
        print(f"  [{s['index']}] {s['source']}{page}  score={s['score']:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lex6", description="Agentic RAG over climate-law documents")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Validate, translate, chunk, embed and store files/directories")
    p.add_argument("paths", nargs="+")
    p.add_argument("--json", action="store_true", help="Also print the reports as JSON")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("ask", help="Answer a question with a notebook RAG pipeline (rag_advanced)")
    p.add_argument("question")
    p.add_argument("--retriever", choices=["naive", "advanced"], default="advanced")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
