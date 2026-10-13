"""``lex6`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from lex6.llm import LLMError


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
    from lex6.services import build_agent, build_llm, build_retriever

    if args.pipeline == "agent":
        response = build_agent().run(args.question)
        if args.json:
            print(json.dumps(response.to_dict(), indent=2, default=str))
            return 0
        print(response.answer)
        for step in response.steps:
            print(f"  · {step.action}: {step.detail}", file=sys.stderr)
        if response.citations:
            print("\nCitations:")
            for c in response.citations:
                page = f" p.{c['page']}" if c["page"] is not None else ""
                print(f"  [{c['index']}] {c['source']}{page} ({c['chunk_id']})")
        return 0

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


def _cmd_eval(args: argparse.Namespace) -> int:
    from lex6.eval import AgentSystem, NaiveBaseline, format_report, load_questions, run_eval
    from lex6.services import build_agent, build_llm, build_retriever, build_vector_store

    questions = load_questions(args.questions)
    if args.only:
        questions = [q for q in questions if q.id in set(args.only.split(","))]
    vs = build_vector_store()
    systems = {
        "baseline": NaiveBaseline(build_retriever("naive", vector_store=vs), build_llm(), top_k=args.top_k),
        "agent": AgentSystem(build_agent(vector_store=vs)),
    }

    def progress(qid: str, name: str, result: dict) -> None:
        print(
            f"  {qid} {name:8s} overall={result['overall']:.2f} {result['output']['latency_s']:.0f}s", file=sys.stderr
        )

    rows = run_eval(questions, systems, on_result=progress)
    print(format_report(rows, list(systems)))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from lex6.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lex6", description="Agentic RAG over climate-law documents")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Validate, translate, chunk, embed and store files/directories")
    p.add_argument("paths", nargs="+")
    p.add_argument("--json", action="store_true", help="Also print the reports as JSON")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("ask", help="Answer a question with the retrieval agent (or --pipeline rag)")
    p.add_argument("question")
    p.add_argument("--pipeline", choices=["agent", "rag"], default="agent", help="agent (default) or rag_advanced")
    p.add_argument("--retriever", choices=["naive", "advanced"], default="advanced", help="for --pipeline rag")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("eval", help="Score the agent against the naive single-shot baseline on a fixed question set")
    p.add_argument("--questions", default=None, help="JSON question set (default: built-in set for data/sample)")
    p.add_argument("--only", default=None, help="comma-separated question ids")
    p.add_argument("--top-k", type=int, default=5, help="baseline top_k")
    p.add_argument("--output", default=None, help="write full per-question results as JSON")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("serve", help="Serve the three.html UI and the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        return args.func(args)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
