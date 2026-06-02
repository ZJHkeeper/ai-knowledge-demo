"""Ingest Markdown files from ./data into a persistent Chroma collection."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION = "ai_knowledge_demo"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class Chunk:
    """A text span ready to be stored in Chroma."""

    text: str
    metadata: dict[str, int | str]


def read_markdown_file(path: Path) -> str:
    """Read Markdown text with common UTF encodings and GB18030 fallback."""

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    joined_errors = "; ".join(errors)
    raise UnicodeDecodeError(
        "markdown",
        b"\x00",
        0,
        1,
        f"failed to decode {path} with utf-8-sig, utf-8, or gb18030 ({joined_errors})",
    )


def discover_markdown_files(data_dir: Path) -> list[Path]:
    """Find Markdown files in a deterministic order."""

    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("*.md"), key=lambda path: path.as_posix().lower())


def chunk_markdown(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split Markdown into Chroma-ready chunks with source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for start, end in _markdown_spans(text):
        chunks.extend(_split_span(text, start, end, source, chunk_size, chunk_overlap))

    return [
        Chunk(
            text=chunk.text,
            metadata={**chunk.metadata, "chunk_index": index},
        )
        for index, chunk in enumerate(chunks)
    ]


def build_chunks_for_files(
    markdown_files: Iterable[Path],
    data_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Chunk], int, list[str]]:
    """Read files and build chunks, returning skipped-file error messages too."""

    all_chunks: list[Chunk] = []
    files_read = 0
    errors: list[str] = []

    for markdown_file in markdown_files:
        source = markdown_file.relative_to(data_dir).as_posix()
        try:
            text = read_markdown_file(markdown_file)
        except UnicodeDecodeError as exc:
            errors.append(str(exc))
            continue

        file_chunks = chunk_markdown(text, source, chunk_size, chunk_overlap)
        all_chunks.extend(file_chunks)
        files_read += 1

    return all_chunks, files_read, errors


def ingest_chunks(chunks: list[Chunk], persist_dir: Path, collection_name: str) -> None:
    """Write chunks into Chroma, replacing chunks from the same source."""

    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name)

    sources = sorted({str(chunk.metadata["source"]) for chunk in chunks})
    for source in sources:
        collection.delete(where={"source": source})

    if not chunks:
        return

    collection.add(
        ids=[_chunk_id(chunk) for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Ingest Markdown files from ./data into a Chroma vector database."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser.parse_args()


def main() -> int:
    """Run Markdown ingestion."""

    args = parse_args()
    data_dir = args.data_dir.resolve()
    persist_dir = args.persist_dir.resolve()

    markdown_files = discover_markdown_files(data_dir)
    chunks, files_read, errors = build_chunks_for_files(
        markdown_files,
        data_dir,
        args.chunk_size,
        args.chunk_overlap,
    )
    ingest_chunks(chunks, persist_dir, args.collection)

    for error in errors:
        print(f"Skipped file: {error}")

    print(f"Files read: {files_read}")
    print(f"Chunks written: {len(chunks)}")
    print(f"Collection: {args.collection}")
    print(f"Chroma path: {persist_dir}")
    return 0


def _markdown_spans(text: str) -> list[tuple[int, int]]:
    boundaries = [0]
    for match in re.finditer(r"(?m)^(?:#{1,6}\s+.+|---\s*)$", text):
        if match.start() not in boundaries:
            boundaries.append(match.start())
    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))

    spans: list[tuple[int, int]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        trimmed = _trim_span(text, start, end)
        if trimmed is not None:
            spans.append(trimmed)
    return spans


def _split_span(
    text: str,
    start: int,
    end: int,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    position = start

    while position < end:
        chunk_end = min(position + chunk_size, end)
        if chunk_end < end:
            chunk_end = _best_break(text, position, chunk_end)

        trimmed = _trim_span(text, position, chunk_end)
        if trimmed is not None:
            trimmed_start, trimmed_end = trimmed
            chunks.append(
                Chunk(
                    text=text[trimmed_start:trimmed_end],
                    metadata={
                        "source": source,
                        "chunk_index": 0,
                        "start_char": trimmed_start,
                        "end_char": trimmed_end,
                    },
                )
            )

        if chunk_end >= end:
            break

        next_position = max(start, chunk_end - chunk_overlap)
        if next_position <= position:
            next_position = chunk_end
        position = next_position

    return chunks


def _best_break(text: str, start: int, default_end: int) -> int:
    minimum = start + max(1, int((default_end - start) * 0.75))
    for marker in ("\n\n", "\n", " "):
        index = text.rfind(marker, minimum, default_end)
        if index != -1:
            return index + len(marker)
    return default_end


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _chunk_id(chunk: Chunk) -> str:
    source = str(chunk.metadata["source"]).replace("\\", "/")
    index = int(chunk.metadata["chunk_index"])
    return f"{source}:{index}"


if __name__ == "__main__":
    raise SystemExit(main())
