"""Ask questions against the local Chroma knowledge base with Ollama."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_knowledge_demo.ingest import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR


DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 4
NO_CONTEXT_MESSAGE = "\u77e5\u8bc6\u5e93\u4e2d\u6ca1\u6709\u627e\u5230\u76f8\u5173\u4fe1\u606f\u3002"


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved Chroma document and its source metadata."""

    text: str
    metadata: Mapping[str, Any]
    distance: float | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Ask a question by retrieving context from Chroma and calling Ollama."
    )
    parser.add_argument("question", help="Question to answer from the knowledge base.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    return parser.parse_args()


def retrieve_chunks(
    question: str,
    persist_dir: Path,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks from a persistent Chroma collection."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=collection_name)
    if collection.count() == 0:
        return []

    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return chunks_from_query_result(result)


def chunks_from_query_result(result: Mapping[str, Any]) -> list[RetrievedChunk]:
    """Convert Chroma's nested query response into a flat chunk list."""

    documents = _first_result_list(result.get("documents"))
    metadatas = _first_result_list(result.get("metadatas"))
    distances = _first_result_list(result.get("distances"))

    chunks: list[RetrievedChunk] = []
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
    return answer


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
                    "\u7684\u77e5\u8bc6\u5e93\u4e0a\u4e0b\u6587\u56de\u7b54\uff1b"
                    f"\u5982\u679c\u4e0a\u4e0b\u6587\u6ca1\u6709\u4f9d\u636e\uff0c"
                    f"\u56de\u7b54\u201c{NO_CONTEXT_MESSAGE}\u201d\u3002"
                    "\u56de\u7b54\u8981\u7b80\u6d01\u3002"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"\u95ee\u9898\uff1a{question}\n\n"
                    f"\u77e5\u8bc6\u5e93\u4e0a\u4e0b\u6587\uff1a\n{context}\n\n"
                    "\u8bf7\u56de\u7b54\u95ee\u9898\u3002"
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

    if not chunks:
        return answer

    source_blocks = "\n\n".join(
        f"- {format_source(chunk.metadata)}\n{chunk.text}" for chunk in chunks
    )
    return f"{answer}\n\n\u6765\u6e90\uff1a\n{source_blocks}"


def main() -> int:
    """Run the RAG question-answering CLI."""

    args = parse_args()

    try:
        chunks = retrieve_chunks(
            question=args.question,
            persist_dir=args.persist_dir.resolve(),
            collection_name=args.collection,
            top_k=args.top_k,
        )
        answer = answer_question(args.question, chunks, args.model, args.ollama_url)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_answer(answer, chunks))
    return 0


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


if __name__ == "__main__":
    raise SystemExit(main())
