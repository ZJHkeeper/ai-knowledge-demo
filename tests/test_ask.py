import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ai_knowledge_demo.ask import (
    DEFAULT_MODEL,
    NO_CONTEXT_MESSAGE,
    RetrievedChunk,
    answer_question,
    build_ollama_payload,
    chunks_from_query_result,
    format_answer,
    format_context,
    format_source,
)


class AskTests(unittest.TestCase):
    def test_chroma_query_result_is_flattened_into_chunks(self) -> None:
        result = {
            "documents": [["first chunk", "second chunk"]],
            "metadatas": [
                [
                    {"source": "refund_policy.md", "chunk_index": 1},
                    {"source": "refund_policy.md", "chunk_index": 2},
                ]
            ],
            "distances": [[0.25, 0.5]],
        }

        chunks = chunks_from_query_result(result)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "first chunk")
        self.assertEqual(chunks[0].metadata["source"], "refund_policy.md")
        self.assertEqual(chunks[0].distance, 0.25)
        self.assertEqual(chunks[1].metadata["chunk_index"], 2)

    def test_empty_query_result_returns_no_chunks(self) -> None:
        chunks = chunks_from_query_result({"documents": [[]], "metadatas": [[]]})

        self.assertEqual(chunks, [])
        self.assertEqual(answer_question("refund time?", chunks, DEFAULT_MODEL), NO_CONTEXT_MESSAGE)

    def test_context_and_answer_include_sources(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Refunds are returned in 3-5 business days after approval.",
                metadata={"source": "refund_policy.md", "chunk_index": 3},
            )
        ]

        self.assertEqual(format_source(chunks[0].metadata), "refund_policy.md#chunk=3")
        self.assertIn("[refund_policy.md#chunk=3]", format_context(chunks))
        self.assertIn(
            "refund_policy.md#chunk=3",
            format_answer("Refunds arrive in 3-5 business days.", chunks),
        )

    def test_ollama_payload_uses_local_model_and_context(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Refunds are returned in 3-5 business days after approval.",
                metadata={"source": "refund_policy.md", "chunk_index": 3},
            )
        ]

        payload = build_ollama_payload("refund time?", chunks, "qwen2.5:7b")

        self.assertEqual(payload["model"], "qwen2.5:7b")
        self.assertFalse(payload["stream"])
        self.assertIn("refund_policy.md#chunk=3", payload["messages"][1]["content"])

    def test_answer_question_calls_ollama_without_openai_key(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Refunds are returned in 3-5 business days after approval.",
                metadata={"source": "refund_policy.md", "chunk_index": 3},
            )
        ]

        with patch(
            "ai_knowledge_demo.ask.post_ollama_chat",
            return_value={"message": {"content": "Refunds arrive in 3-5 business days."}},
        ) as post_ollama_chat:
            answer = answer_question("refund time?", chunks, "qwen2.5:7b")

        self.assertEqual(answer, "Refunds arrive in 3-5 business days.")
        post_ollama_chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
