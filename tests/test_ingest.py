import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ai_knowledge_demo.ingest import chunk_markdown


class IngestChunkTests(unittest.TestCase):
    def test_short_markdown_stays_in_one_chunk(self) -> None:
        text = "# Refunds\n\nRefund requests must be filed within 7 days."

        chunks = chunk_markdown(text, "refund_policy.md", chunk_size=800, chunk_overlap=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].metadata["source"], "refund_policy.md")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].metadata["start_char"], 0)
        self.assertEqual(chunks[0].metadata["end_char"], len(text))

    def test_long_paragraph_uses_size_and_overlap(self) -> None:
        text = "a" * 25

        chunks = chunk_markdown(text, "long.md", chunk_size=10, chunk_overlap=3)

        self.assertEqual([chunk.text for chunk in chunks], ["a" * 10, "a" * 10, "a" * 10, "a" * 4])
        self.assertEqual(
            [(chunk.metadata["start_char"], chunk.metadata["end_char"]) for chunk in chunks],
            [(0, 10), (7, 17), (14, 24), (21, 25)],
        )

    def test_metadata_is_stable_across_markdown_sections(self) -> None:
        text = "# One\n\nFirst section.\n\n---\n\n## Two\n\nSecond section."

        chunks = chunk_markdown(text, "nested/refund_policy.md", chunk_size=800, chunk_overlap=100)

        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], [0, 1])
        self.assertEqual(
            [chunk.metadata["source"] for chunk in chunks],
            ["nested/refund_policy.md", "nested/refund_policy.md"],
        )
        self.assertEqual(chunks[0].text, "# One\n\nFirst section.")
        self.assertEqual(chunks[1].text, "## Two\n\nSecond section.")

    def test_thematic_breaks_only_split_sections(self) -> None:
        text = "# One\n\nFirst\n\n***\n\n## Two\n\nSecond\n\n___\n\n## Three\n\nThird\n\n- - -\n\n## Four\n\nFourth"

        chunks = chunk_markdown(text, "breaks.md", chunk_size=800, chunk_overlap=100)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "# One\n\nFirst",
                "## Two\n\nSecond",
                "## Three\n\nThird",
                "## Four\n\nFourth",
            ],
        )
        self.assertFalse(any(chunk.text.strip() in {"---", "***", "___", "- - -"} for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
