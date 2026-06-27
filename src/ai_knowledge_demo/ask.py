"""Ask questions against the local Chroma knowledge base with Ollama."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_knowledge_demo.ingest import (
    BM25_TOKEN_EXPANSIONS,
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    default_bm25_index_path,
    tokenize_for_bm25,
)


DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 4
DEFAULT_MIN_RECALL = 20
DEFAULT_RECALL_MULTIPLIER = 5
DEFAULT_QUERY_REWRITE_COUNT = 4
VECTOR_SCORE_WEIGHT = 2.0
BM25_SCORE_WEIGHT = 10.0
NO_CONTEXT_MESSAGE = "\u77e5\u8bc6\u5e93\u4e2d\u6ca1\u6709\u627e\u5230\u76f8\u5173\u4fe1\u606f\u3002"
CONCLUSION_LABEL = "\u7ed3\u8bba\uff1a"
CONFIRMED_LABEL = "\u53ef\u4ee5\u786e\u8ba4\uff1a"
UNCLEAR_LABEL = "\u672a\u660e\u786e\u8bf4\u660e\uff1a"
RELATED_BUT_UNCLEAR_MESSAGE = (
    "\u77e5\u8bc6\u5e93\u4e2d\u6709\u76f8\u5173\u4fe1\u606f\uff0c"
    "\u4f46\u672a\u660e\u786e\u8bf4\u660e\u7528\u6237\u95ee\u9898\u7684\u5177\u4f53\u7b54\u6848\u3002"
)
RELATED_CONTEXT_FALLBACK = (
    "\u77e5\u8bc6\u5e93\u5305\u542b\u4e0e\u95ee\u9898\u76f8\u5173\u7684\u5185\u5bb9\uff0c"
    "\u8be6\u89c1\u4e0b\u65b9\u6765\u6e90\u3002"
)
LOW_SIGNAL_TERMS = {
    "app",
    "\u53ef\u4ee5",
    "\u5546\u54c1",
    "\u5904\u7406",
    "\u600e\u4e48",
    "\u663e\u793a",
    "\u7528\u6237",
    "\u95ee\u9898",
    "\u8bf4\u660e",
}
HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved Chroma document and its source metadata."""

    text: str
    metadata: Mapping[str, Any]
    distance: float | None = None
    bm25_score: float | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Ask a question by retrieving context from Chroma and calling Ollama."
    )
    parser.add_argument("question", help="Question to answer from the knowledge base.")
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
    return parser.parse_args()


def retrieve_chunks(
    question: str,
    persist_dir: Path,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
    search_queries: list[str] | None = None,
    bm25_index_path: Path | None = None,
    use_bm25: bool = True,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks with Chroma vector search and optional BM25."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)
    collection_count = collection.count()
    if collection_count == 0:
        return []

    candidate_count = min(
        collection_count,
        max(top_k * DEFAULT_RECALL_MULTIPLIER, DEFAULT_MIN_RECALL),
    )
    queries = search_queries or [question]
    result = collection.query(
        query_texts=queries,
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    candidates = chunks_from_query_result(result)
    if use_bm25:
        bm25_path = bm25_index_path or default_bm25_index_path(persist_dir)
        candidates.extend(retrieve_bm25_chunks(queries, bm25_path, candidate_count))

    candidates = dedupe_chunks(candidates)
    return rerank_chunks(question, candidates, top_k, scoring_queries=queries)


def retrieve_bm25_chunks(
    queries: list[str],
    index_path: Path,
    candidate_count: int,
) -> list[RetrievedChunk]:
    """Retrieve keyword-matched chunks from the persisted BM25 index."""

    if candidate_count <= 0 or not index_path.exists():
        return []

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    entries = [
        entry
        for entry in index.get("chunks", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("text"), str)
        and isinstance(entry.get("tokens"), list)
    ]
    if not entries:
        return []

    from rank_bm25 import BM25Okapi

    corpus = [[str(token) for token in entry["tokens"]] for entry in entries]
    bm25 = BM25Okapi(corpus)
    scores = [0.0 for _entry in entries]
    for query in queries:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            continue
        for index, score in enumerate(bm25.get_scores(query_tokens)):
            scores[index] += float(score)

    ranked_indexes = sorted(
        (index for index, score in enumerate(scores) if score > 0),
        key=lambda index: (-scores[index], index),
    )[:candidate_count]

    chunks: list[RetrievedChunk] = []
    for index in ranked_indexes:
        entry = entries[index]
        metadata = entry.get("metadata")
        chunks.append(
            RetrievedChunk(
                text=str(entry["text"]),
                metadata=metadata if isinstance(metadata, dict) else {},
                bm25_score=scores[index],
            )
        )
    return chunks


def chunks_from_query_result(result: Mapping[str, Any]) -> list[RetrievedChunk]:
    """Convert Chroma's nested query response into a flat chunk list."""

    chunks: list[RetrievedChunk] = []
    document_groups = _result_groups(result.get("documents"))
    metadata_groups = _result_groups(result.get("metadatas"))
    distance_groups = _result_groups(result.get("distances"))

    for group_index, documents in enumerate(document_groups):
        metadatas = metadata_groups[group_index] if group_index < len(metadata_groups) else []
        distances = distance_groups[group_index] if group_index < len(distance_groups) else []
        for index, document in enumerate(documents):
            if not document:
                continue

            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            distance = distances[index] if index < len(distances) else None
            chunks.append(
                RetrievedChunk(
                    text=str(document),
                    metadata=metadata,
                    distance=float(distance) if distance is not None else None,
                )
            )

    return chunks


def dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate chunks, keeping best vector distance and BM25 score."""

    by_key: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for chunk in chunks:
        key = format_source(chunk.metadata)
        if key == "unknown":
            key = chunk.text

        existing = by_key.get(key)
        if existing is None:
            by_key[key] = chunk
            order.append(key)
            continue

        by_key[key] = merge_retrieved_chunks(existing, chunk)

    return [by_key[key] for key in order]


def merge_retrieved_chunks(existing: RetrievedChunk, chunk: RetrievedChunk) -> RetrievedChunk:
    """Merge duplicate retrieval hits from different retrieval methods."""

    existing_distance = existing.distance if existing.distance is not None else float("inf")
    chunk_distance = chunk.distance if chunk.distance is not None else float("inf")
    distance = min(existing_distance, chunk_distance)
    if distance == float("inf"):
        distance = None

    bm25_score = max(existing.bm25_score or 0.0, chunk.bm25_score or 0.0)
    best_chunk = chunk if chunk_distance < existing_distance else existing
    return RetrievedChunk(
        text=best_chunk.text,
        metadata=best_chunk.metadata,
        distance=distance,
        bm25_score=bm25_score or None,
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as source-labeled context for the model."""

    return "\n\n".join(
        f"[{format_source(chunk.metadata)}]\n{chunk.text}" for chunk in chunks
    )


def format_source(metadata: Mapping[str, Any]) -> str:
    """Format Chroma metadata into a stable source label."""

    source = str(metadata.get("source") or "unknown")
    chunk_index = metadata.get("chunk_index")
    if chunk_index is None:
        return source
    return f"{source}#chunk={chunk_index}"


def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    scoring_queries: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Rerank hybrid candidates with keyword, BM25, and vector signals."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not chunks:
        return []

    queries = scoring_queries or [question]
    max_bm25_score = max((chunk.bm25_score or 0.0 for chunk in chunks), default=0.0)
    scored_chunks = [
        (
            -hybrid_score(chunk, queries, max_bm25_score),
            chunk.distance if chunk.distance is not None else float("inf"),
            index,
            chunk,
        )
        for index, chunk in enumerate(chunks)
    ]
    ranked_chunks = [chunk for *_unused, chunk in sorted(scored_chunks)]
    selected_chunks = ranked_chunks[:top_k]
    if max_bm25_score > 0 and top_k > 1:
        bm25_leader = max(
            chunks,
            key=lambda chunk: (chunk.bm25_score or 0.0, -chunks.index(chunk)),
        )
        if bm25_leader not in selected_chunks:
            selected_chunks = [
                bm25_leader,
                *[chunk for chunk in selected_chunks if chunk is not bm25_leader],
            ][:top_k]

    return selected_chunks


def hybrid_score(
    chunk: RetrievedChunk,
    queries: list[str],
    max_bm25_score: float,
) -> float:
    """Combine exact keyword, BM25, and vector distance into one score."""

    keyword_total = sum(keyword_score(query, chunk) for query in queries)
    bm25_score = chunk.bm25_score or 0.0
    normalized_bm25 = bm25_score / max_bm25_score if max_bm25_score > 0 else 0.0
    return (
        keyword_total
        + normalized_bm25 * BM25_SCORE_WEIGHT
        + vector_weight_score(chunk)
    )


def vector_weight_score(chunk: RetrievedChunk) -> float:
    """Return the weighted vector contribution used by hybrid reranking."""

    if chunk.distance is None:
        return 0.0
    return (1.0 / (1.0 + chunk.distance)) * VECTOR_SCORE_WEIGHT


def keyword_weight_score(chunk: RetrievedChunk, max_bm25_score: float) -> float:
    """Return the weighted BM25 keyword contribution used by hybrid reranking."""

    if max_bm25_score <= 0:
        return 0.0
    return ((chunk.bm25_score or 0.0) / max_bm25_score) * BM25_SCORE_WEIGHT


def keyword_score(question: str, chunk: RetrievedChunk) -> float:
    """Score how well a chunk matches question keywords."""

    terms = extract_query_terms(question)
    if not terms:
        return 0.0

    title = _first_heading(chunk.text)
    title_lower = title.lower()
    body_lower = chunk.text.lower()
    score = 0.0

    for term in terms:
        term_lower = term.lower()
        weight = len(term_lower)
        if term_lower in title_lower:
            score += 6.0 + weight
        if term_lower in body_lower:
            score += 1.5 + weight / 2.0

    compact_question = _compact_text(question)
    compact_text = _compact_text(chunk.text)
    if len(compact_question) >= 4 and compact_question in compact_text:
        score += 12.0

    return score


def extract_query_terms(question: str) -> list[str]:
    """Extract Chinese n-grams and alphanumeric keywords from a question."""

    terms: set[str] = set()
    for match in re.finditer(r"[A-Za-z0-9]+", question):
        value = match.group(0).lower()
        if len(value) >= 2:
            terms.add(value)
            terms.update(BM25_TOKEN_EXPANSIONS.get(value, ()))

    for match in re.finditer(r"[\u4e00-\u9fff]+", question):
        value = match.group(0)
        if len(value) >= 2:
            terms.add(value)
        for size in (2, 3):
            if len(value) >= size:
                for index in range(len(value) - size + 1):
                    terms.add(value[index : index + size])

    terms = {term for term in terms if term not in LOW_SIGNAL_TERMS}
    return sorted(terms, key=lambda term: (-len(term), term))


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Call Ollama to answer the question using retrieved chunks."""

    if not chunks:
        return NO_CONTEXT_MESSAGE

    payload = build_ollama_payload(question, chunks, model)
    response = post_ollama_chat(ollama_url, payload)
    answer = response.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")
    return normalize_answer_template(answer, question, chunks)


def generate_search_queries(
    question: str,
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    rewrite_count: int = DEFAULT_QUERY_REWRITE_COUNT,
) -> list[str]:
    """Generate retrieval-focused query rewrites with Ollama."""

    if rewrite_count <= 0:
        return [question]

    payload = build_query_rewrite_payload(question, model, rewrite_count)
    try:
        response = post_ollama_chat(ollama_url, payload)
    except RuntimeError:
        return [question]

    content = response.get("message", {}).get("content", "")
    rewrites = parse_search_queries(str(content), rewrite_count)
    queries = [question, *rewrites]
    return _unique_nonempty(queries)


def build_query_rewrite_payload(
    question: str,
    model: str,
    rewrite_count: int = DEFAULT_QUERY_REWRITE_COUNT,
) -> dict[str, Any]:
    """Build an Ollama request that rewrites the user question for retrieval."""

    return {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "\u4f60\u662f RAG \u68c0\u7d22\u67e5\u8be2\u6539\u5199\u5668\u3002"
                    "\u8bf7\u628a\u7528\u6237\u95ee\u9898\u6539\u5199\u6210\u66f4\u9002\u5408"
                    "\u77e5\u8bc6\u5e93\u68c0\u7d22\u7684\u77ed\u67e5\u8be2\u3002"
                    "\u8981\u4f7f\u7528\u591a\u79cd\u8868\u8fbe\uff0c\u8986\u76d6\u540c\u4e49\u8bf4\u6cd5\u3001"
                    "\u4e1a\u52a1\u672f\u8bed\u3001\u4e0a\u4f4d\u7c7b\u76ee\u548c\u53ef\u80fd\u7684\u5904\u7406\u65b9\u5f0f\u3002"
                    "\u4e0d\u8981\u53ea\u6539\u5199\u5b57\u9762\u8868\u8fbe\uff0c"
                    "\u8981\u628a\u7528\u6237\u7684\u53e3\u8bed\u573a\u666f\u8f6c\u6210\u77e5\u8bc6\u5e93\u53ef\u80fd\u4f7f\u7528\u7684\u653f\u7b56\u8bcd\u3002"
                    "\u4f8b\u5982\uff1a\u201cApp \u663e\u793a\u5df2\u9001\u8fbe\u4f46\u6ca1\u6536\u5230\u5546\u54c1\u201d"
                    "\u5e94\u6269\u5c55\u4e3a\u7269\u6d41\u5f02\u5e38\u3001\u8fd0\u8f93\u4e22\u5931\u3001"
                    "\u672a\u6536\u8d27\u3001\u5ba2\u670d\u3001\u8865\u53d1\u3001\u5168\u989d\u9000\u6b3e\u7b49\u68c0\u7d22\u8bcd\u3002"
                    "\u4f8b\u5982\uff1a\u201c\u4e2a\u4eba\u53d1\u7968\u80fd\u5426\u5f00\u5177\u201d"
                    "\u5e94\u6269\u5c55\u4e3a\u53d1\u7968\u3001\u7535\u5b50\u53d1\u7968\u3001"
                    "\u589e\u503c\u7a0e\u4e13\u7528\u53d1\u7968\u3001\u4e2a\u4eba\u7528\u6237\u3001\u4f01\u4e1a\u7528\u6237\u7b49\u68c0\u7d22\u8bcd\u3002"
                    "\u53ea\u8f93\u51fa JSON \u6570\u7ec4\uff0c\u6570\u7ec4\u5143\u7d20\u662f\u5b57\u7b26\u4e32\uff0c"
                    "\u4e0d\u8981\u8f93\u51fa\u5176\u4ed6\u5185\u5bb9\u3002"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"\u8bf7\u751f\u6210 {rewrite_count} \u6761\u68c0\u7d22\u67e5\u8be2\u6539\u5199\uff1a"
                    f"{question}"
                ),
            },
        ],
    }


def parse_search_queries(content: str, limit: int) -> list[str]:
    """Parse query rewrites from JSON or line-oriented model output."""

    candidates: list[str] = []
    json_text = _extract_json_array(content)
    if json_text is not None:
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            candidates.extend(str(item) for item in parsed)

    if not candidates:
        for line in content.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                candidates.append(cleaned.strip("\"'"))

    return _unique_nonempty(candidates)[:limit]


def normalize_answer_template(
    answer: str,
    question: str,
    chunks: list[RetrievedChunk],
) -> str:
    """Keep local-model answers inside the fixed answerability template."""

    normalized = answer.strip()
    if not normalized:
        return normalized

    has_related_context = any(keyword_score(question, chunk) > 0 for chunk in chunks)
    if has_related_context and normalized.startswith(f"{CONCLUSION_LABEL}{NO_CONTEXT_MESSAGE}"):
        normalized = normalized.replace(
            f"{CONCLUSION_LABEL}{NO_CONTEXT_MESSAGE}",
            f"{CONCLUSION_LABEL}{RELATED_BUT_UNCLEAR_MESSAGE}",
            1,
        )

    if CONCLUSION_LABEL not in normalized:
        normalized = f"{CONCLUSION_LABEL}{RELATED_BUT_UNCLEAR_MESSAGE}\n{normalized}"

    if CONFIRMED_LABEL not in normalized:
        confirmed = RELATED_CONTEXT_FALLBACK if has_related_context else "\u65e0"
        normalized = f"{normalized}\n{CONFIRMED_LABEL}\n- {confirmed}"
    elif has_related_context:
        normalized = _replace_empty_section(
            normalized,
            CONFIRMED_LABEL,
            UNCLEAR_LABEL,
            RELATED_CONTEXT_FALLBACK,
        )

    if UNCLEAR_LABEL not in normalized:
        normalized = f"{normalized}\n{UNCLEAR_LABEL}\n- \u65e0"

    return normalized


def build_ollama_payload(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
) -> dict[str, Any]:
    """Build the Ollama /api/chat request payload."""

    context = format_context(chunks)
    return {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "\u4f60\u662f\u4e00\u4e2a\u4e25\u8c28\u7684\u4e2d\u6587\u77e5\u8bc6\u5e93"
                    "\u95ee\u7b54\u52a9\u624b\u3002\u53ea\u80fd\u6839\u636e\u63d0\u4f9b"
                    "\u7684\u77e5\u8bc6\u5e93\u4e0a\u4e0b\u6587\u56de\u7b54\u3002"
                    "\u5148\u5224\u65ad\u95ee\u9898\u7684 answerability\uff0c"
                    "\u53ea\u80fd\u662f\u4ee5\u4e0b\u4e09\u6863\u4e4b\u4e00\uff1a"
                    "\u53ef\u76f4\u63a5\u56de\u7b54\uff1a\u4e0a\u4e0b\u6587\u660e\u786e\u56de\u7b54\u7528\u6237\u95ee\u9898\u3002"
                    "\u90e8\u5206\u76f8\u5173\u4f46\u672a\u660e\u786e\uff1a"
                    "\u4e0a\u4e0b\u6587\u4e0e\u95ee\u9898\u76f8\u5173\uff0c"
                    "\u4f46\u6ca1\u6709\u8986\u76d6\u7528\u6237\u8ffd\u95ee\u7684\u5177\u4f53\u70b9\u3002"
                    "\u5b8c\u5168\u65e0\u5173\uff1a\u4e0a\u4e0b\u6587\u548c\u95ee\u9898\u65e0\u5173\u3002"
                    "\u5982\u679c\u4e0a\u4e0b\u6587\u548c\u95ee\u9898\u5171\u4eab\u540c\u4e00\u4e1a\u52a1\u4e3b\u9898"
                    "\uff08\u4f8b\u5982\u90fd\u5728\u8bf4\u53d1\u7968\u3001\u9000\u6b3e\u3001\u7269\u6d41\u7b49\uff09\uff0c"
                    "\u5373\u4f7f\u6ca1\u6709\u76f4\u63a5\u7ed3\u8bba\uff0c"
                    "\u4e5f\u5fc5\u987b\u5224\u4e3a\u201c\u90e8\u5206\u76f8\u5173\u4f46\u672a\u660e\u786e\u201d\uff0c"
                    "\u4e0d\u80fd\u5224\u4e3a\u201c\u5b8c\u5168\u65e0\u5173\u201d\u3002"
                    "\u53ea\u8981\u4f60\u80fd\u4ece\u4e0a\u4e0b\u6587\u5217\u51fa\u4efb\u4f55\u4e00\u6761"
                    "\u4e0e\u95ee\u9898\u76f8\u5173\u7684\u53ef\u786e\u8ba4\u4e8b\u5b9e\uff0c"
                    f"\u7ed3\u8bba\u5c31\u4e0d\u80fd\u4f7f\u7528\u201c{NO_CONTEXT_MESSAGE}\u201d\u3002"
                    "\u5fc5\u987b\u4e14\u53ea\u80fd\u4f7f\u7528\u4e0b\u9762\u7684\u56fa\u5b9a\u8f93\u51fa\u683c\u5f0f\uff0c"
                    "\u4e0d\u8981\u8f93\u51fa\u6a21\u677f\u5916\u7684\u5176\u4ed6\u6bb5\u843d\u3002"
                    "\u5373\u4f7f\u67d0\u4e2a\u5b57\u6bb5\u5185\u5bb9\u4e3a\u201c\u65e0\u201d\uff0c"
                    "\u4e5f\u5fc5\u987b\u4fdd\u7559\u8be5\u5b57\u6bb5\u6807\u9898\u3002"
                    "\u56fa\u5b9a\u8f93\u51fa\u683c\u5f0f\uff1a"
                    "\u7ed3\u8bba\uff1a...\n"
                    "\u53ef\u4ee5\u786e\u8ba4\uff1a\n"
                    "- ...\n"
                    "\u672a\u660e\u786e\u8bf4\u660e\uff1a\n"
                    "- ...\n"
                    "\u5bf9\u4e8e\u201c\u53ef\u76f4\u63a5\u56de\u7b54\u201d\uff0c"
                    "\u7ed3\u8bba\u76f4\u63a5\u56de\u7b54\uff1b\u53ef\u4ee5\u786e\u8ba4\u5217\u51fa 1-3 \u6761\u4f9d\u636e\uff1b"
                    "\u672a\u660e\u786e\u8bf4\u660e\u5199\u201c\u65e0\u201d\u3002"
                    "\u5bf9\u4e8e\u201c\u90e8\u5206\u76f8\u5173\u4f46\u672a\u660e\u786e\u201d\uff0c"
                    "\u7ed3\u8bba\u8bf4\u660e\u77e5\u8bc6\u5e93\u4e2d\u6709\u76f8\u5173\u4fe1\u606f\uff0c"
                    "\u4f46\u672a\u660e\u786e\u8bf4\u660e\u7528\u6237\u8ffd\u95ee\u7684\u5177\u4f53\u70b9\uff1b"
                    "\u53ef\u4ee5\u786e\u8ba4\u5217\u51fa 1-3 \u6761\u76f8\u5173\u4e8b\u5b9e\uff1b"
                    "\u672a\u660e\u786e\u8bf4\u660e\u5217\u51fa 1-3 \u6761\u7f3a\u53e3\u3002"
                    "\u5bf9\u4e8e\u201c\u5b8c\u5168\u65e0\u5173\u201d\uff0c"
                    f"\u7ed3\u8bba\u5fc5\u987b\u662f\u201c{NO_CONTEXT_MESSAGE}\u201d\uff1b"
                    "\u53ef\u4ee5\u786e\u8ba4\u5199\u201c\u65e0\u201d\uff1b"
                    "\u672a\u660e\u786e\u8bf4\u660e\u5199\u201c\u7528\u6237\u95ee\u9898\u76f8\u5173\u5185\u5bb9\u672a\u5728\u77e5\u8bc6\u5e93\u4e2d\u51fa\u73b0\u201d\u3002"
                    "\u56de\u7b54\u8981\u7b80\u6d01\uff0c\u4e0d\u8981\u7f16\u9020\u77e5\u8bc6\u5e93\u4e2d\u6ca1\u6709\u7684\u7ed3\u8bba\u3002"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"\u95ee\u9898\uff1a{question}\n\n"
                    f"\u77e5\u8bc6\u5e93\u4e0a\u4e0b\u6587\uff1a\n{context}\n\n"
                    "\u8bf7\u4e25\u683c\u6309\u7cfb\u7edf\u6d88\u606f\u4e2d\u7684"
                    "\u56fa\u5b9a\u6a21\u677f\u56de\u7b54\uff0c\u4e0d\u8981\u8f93\u51fa\u6a21\u677f\u5916\u5185\u5bb9\u3002"
                ),
            },
        ],
    }


def post_ollama_chat(ollama_url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Post a non-streaming chat request to Ollama."""

    endpoint = f"{ollama_url.rstrip('/')}/api/chat"
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Start Ollama first, then run "
            f"`ollama pull {payload.get('model', DEFAULT_MODEL)}` if the model is missing."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc


def format_answer(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Append source labels and chunk text to an answer."""

    return format_answer_with_queries(answer, chunks)


def format_answer_with_queries(
    answer: str,
    chunks: list[RetrievedChunk],
    search_queries: list[str] | None = None,
) -> str:
    """Format query rewrites, answer, and source chunks for CLI output."""

    output_parts: list[str] = []
    if search_queries:
        output_parts.append(format_search_queries(search_queries))

    output_parts.append(answer)

    if not chunks:
        return "\n\n".join(output_parts)

    max_bm25_score = max((chunk.bm25_score or 0.0 for chunk in chunks), default=0.0)
    source_blocks = "\n\n".join(
        f"- {format_chunk_source_details(chunk, max_bm25_score)}\n{chunk.text}"
        for chunk in chunks
    )
    output_parts.append(f"\u6765\u6e90\uff1a\n{source_blocks}")
    return "\n\n".join(output_parts)


def format_chunk_source_details(chunk: RetrievedChunk, max_bm25_score: float) -> str:
    """Format source and retrieval score details for CLI output."""

    return (
        f"{format_source(chunk.metadata)} | "
        f"\u5411\u91cf\u6743\u91cd\u5206={vector_weight_score(chunk):.3f} | "
        f"\u5173\u952e\u8bcd\u6743\u91cd\u5206={keyword_weight_score(chunk, max_bm25_score):.3f}"
    )


def format_search_queries(search_queries: list[str]) -> str:
    """Format retrieval queries for debugging the retrieval step."""

    query_lines = "\n".join(f"- {query}" for query in search_queries)
    return f"\u68c0\u7d22\u67e5\u8be2\uff1a\n{query_lines}"


def main() -> int:
    """Run the RAG question-answering CLI."""

    args = parse_args()

    try:
        search_queries = generate_search_queries(
            args.question,
            args.model,
            args.ollama_url,
        )
        chunks = retrieve_chunks(
            question=args.question,
            persist_dir=args.persist_dir.resolve(),
            collection_name=args.collection,
            top_k=args.top_k,
            search_queries=search_queries,
            bm25_index_path=(
                args.bm25_index.resolve()
                if args.bm25_index is not None
                else None
            ),
            use_bm25=not args.no_bm25,
        )
        answer = answer_question(args.question, chunks, args.model, args.ollama_url)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_answer_with_queries(answer, chunks, search_queries))
    return 0


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _result_groups(value: Any) -> list[list[Any]]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value
    if isinstance(value, list):
        return [value]
    return []


def _first_heading(text: str) -> str:
    match = HEADING_RE.search(text)
    if match is None:
        return ""
    return match.group(1)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _replace_empty_section(
    text: str,
    label: str,
    next_label: str,
    replacement: str,
) -> str:
    start = text.find(label)
    if start == -1:
        return text

    content_start = start + len(label)
    end = text.find(next_label, content_start)
    if end == -1:
        end = len(text)

    section_content = text[content_start:end].strip()
    compact_content = _compact_text(section_content)
    if compact_content not in {"", "-\u65e0", "\u65e0", "\u65e0\u3002"}:
        return text

    return f"{text[:content_start]}\n- {replacement}\n{text[end:]}"


def _extract_json_array(text: str) -> str | None:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(normalized)
    return unique_values


if __name__ == "__main__":
    raise SystemExit(main())
