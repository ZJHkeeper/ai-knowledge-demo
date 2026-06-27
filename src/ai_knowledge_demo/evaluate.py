"""Evaluate retrieval and answer quality for the local RAG demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_knowledge_demo.ask import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TOP_K,
    NO_CONTEXT_MESSAGE,
    answer_question,
    format_source,
    generate_search_queries,
    retrieve_chunks,
)
from ai_knowledge_demo.ingest import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = PROJECT_ROOT / "tests" / "eval_cases.json"


@dataclass(frozen=True)
class EvalResult:
    """The outcome for one eval case."""

    case_id: str
    passed: bool
    failures: list[str]
    sources: list[str]
    answer: str | None = None
    warnings: list[str] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and answers against JSON eval cases."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=None,
        help="Path to the persisted BM25 index. Defaults to <persist-dir>/bm25_index.json.",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Disable BM25 keyword retrieval and use vector retrieval only.",
    )
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only validate retrieval expectations. Query rewrite still runs by default.",
    )
    parser.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Disable query rewrite and retrieve with the original question only.",
    )
    return parser.parse_args(argv)


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate eval cases from JSON."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of eval cases.")

    cases: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Eval case at index {index} must be an object.")
        errors = validate_case(item)
        if errors:
            joined_errors = "; ".join(errors)
            raise ValueError(f"Invalid eval case at index {index}: {joined_errors}")
        cases.append(item)
    return cases


def validate_case(case: Mapping[str, Any]) -> list[str]:
    """Return schema validation errors for one eval case."""

    errors: list[str] = []
    for key in ("id", "type", "type_description", "question"):
        if not isinstance(case.get(key), str) or not str(case.get(key)).strip():
            errors.append(f"{key} must be a non-empty string")

    expected_sources = case.get("expected_sources")
    if not isinstance(expected_sources, list) or not all(
        isinstance(source, str) for source in expected_sources
    ):
        errors.append("expected_sources must be a list of strings")

    for key in ("must_include", "should_include"):
        value = case.get(key, [])
        if not isinstance(value, list) or not all(isinstance(term, str) for term in value):
            errors.append(f"{key} must be a list of strings")

    if not isinstance(case.get("expected_refusal"), bool):
        errors.append("expected_refusal must be a boolean")

    return errors


def evaluate_cases(
    cases: list[Mapping[str, Any]],
    *,
    persist_dir: Path,
    collection_name: str,
    top_k: int,
    bm25_index_path: Path | None,
    use_bm25: bool,
    model: str,
    ollama_url: str,
    retrieval_only: bool,
    use_query_rewrite: bool,
) -> list[EvalResult]:
    """Evaluate all cases and return per-case results."""

    return [
        evaluate_case(
            case,
            persist_dir=persist_dir,
            collection_name=collection_name,
            top_k=top_k,
            bm25_index_path=bm25_index_path,
            use_bm25=use_bm25,
            model=model,
            ollama_url=ollama_url,
            retrieval_only=retrieval_only,
            use_query_rewrite=use_query_rewrite,
        )
        for case in cases
    ]


def evaluate_case(
    case: Mapping[str, Any],
    *,
    persist_dir: Path,
    collection_name: str,
    top_k: int,
    bm25_index_path: Path | None,
    use_bm25: bool,
    model: str,
    ollama_url: str,
    retrieval_only: bool,
    use_query_rewrite: bool,
) -> EvalResult:
    """Evaluate one case against retrieval and, optionally, generated answer text."""

    question = str(case["question"])
    search_queries = (
        generate_search_queries(question, model, ollama_url)
        if use_query_rewrite
        else [question]
    )
    chunks = retrieve_chunks(
        question=question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        top_k=top_k,
        search_queries=search_queries,
        bm25_index_path=bm25_index_path,
        use_bm25=use_bm25,
    )

    failures = check_retrieval(chunks, case.get("expected_sources", []))
    warnings: list[str] = []
    answer: str | None = None
    if not retrieval_only:
        answer = answer_question(question, chunks, model, ollama_url)
        failures.extend(
            check_answer(
                answer,
                must_include=case.get("must_include", []),
                expected_refusal=bool(case["expected_refusal"]),
            )
        )
        warnings.extend(check_should_include(answer, case.get("should_include", [])))

    sources = [format_source(chunk.metadata) for chunk in chunks]
    return EvalResult(
        case_id=str(case["id"]),
        passed=not failures,
        failures=failures,
        sources=sources,
        answer=answer,
        warnings=warnings,
    )


def check_retrieval(
    chunks: Sequence[Any],
    expected_sources: Any,
) -> list[str]:
    """Check that at least one acceptable expected source is present."""

    failures: list[str] = []
    if not expected_sources:
        return failures

    retrieved_sources = {source_from_chunk(chunk) for chunk in chunks}
    expected_source_set = {str(source) for source in expected_sources}
    if retrieved_sources.isdisjoint(expected_source_set):
        expected_text = ", ".join(sorted(expected_source_set))
        failures.append(f"retrieval missing expected source; expected any of: {expected_text}")

    return failures


def source_from_chunk(chunk: Any) -> str:
    """Return the source filename from a retrieved chunk."""

    metadata = getattr(chunk, "metadata", {})
    return str(metadata.get("source") or "")


def check_answer(
    answer: str,
    *,
    must_include: Any,
    expected_refusal: bool,
) -> list[str]:
    """Check hard answer requirements."""

    failures: list[str] = []
    answer_lower = answer.lower()

    for term in must_include:
        term_text = str(term)
        if term_text.lower() not in answer_lower:
            failures.append(f"answer missing required term={term_text}")

    if not expected_refusal and NO_CONTEXT_MESSAGE in answer:
        failures.append("answer unexpectedly refused")

    return failures


def check_should_include(answer: str, should_include: Any) -> list[str]:
    """Check soft answer terms and return warnings."""

    warnings: list[str] = []
    answer_lower = answer.lower()
    for term in should_include:
        term_text = str(term)
        if term_text.lower() not in answer_lower:
            warnings.append(f"answer missing suggested term={term_text}")

    return warnings


def format_results(results: Sequence[EvalResult]) -> str:
    """Format evaluation results for terminal output."""

    lines: list[str] = []
    passed_count = sum(1 for result in results if result.passed)
    total_count = len(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        sources = ", ".join(result.sources) if result.sources else "<none>"
        lines.append(f"[{status}] {result.case_id}")
        lines.append(f"  sources: {sources}")
        for failure in result.failures:
            lines.append(f"  - {failure}")
        for warning in result.warnings or []:
            lines.append(f"  ! {warning}")

    rate = (passed_count / total_count * 100.0) if total_count else 0.0
    lines.append(f"Summary: {passed_count}/{total_count} passed ({rate:.1f}%).")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation CLI."""

    args = parse_args(argv)
    try:
        cases = load_cases(args.cases)
        results = evaluate_cases(
            cases,
            persist_dir=args.persist_dir.resolve(),
            collection_name=args.collection,
            top_k=args.top_k,
            bm25_index_path=args.bm25_index.resolve() if args.bm25_index is not None else None,
            use_bm25=not args.no_bm25,
            model=args.model,
            ollama_url=args.ollama_url,
            retrieval_only=args.retrieval_only,
            use_query_rewrite=not args.no_query_rewrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(format_results(results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
