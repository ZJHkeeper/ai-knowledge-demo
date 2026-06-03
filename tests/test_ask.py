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
    dedupe_chunks,
    format_answer,
    format_answer_with_queries,
    format_context,
    format_source,
    generate_search_queries,
    normalize_answer_template,
    parse_search_queries,
    rerank_chunks,
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

    def test_chroma_query_result_flattens_multiple_query_groups(self) -> None:
        result = {
            "documents": [["first chunk"], ["second chunk"]],
            "metadatas": [
                [{"source": "refund_policy.md", "chunk_index": 1}],
                [{"source": "refund_policy.md", "chunk_index": 2}],
            ],
            "distances": [[0.25], [0.5]],
        }

        chunks = chunks_from_query_result(result)

        self.assertEqual([chunk.text for chunk in chunks], ["first chunk", "second chunk"])
        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], [1, 2])

    def test_dedupe_chunks_keeps_best_distance(self) -> None:
        chunks = [
            RetrievedChunk(
                text="older duplicate",
                metadata={"source": "refund_policy.md", "chunk_index": 4},
                distance=0.9,
            ),
            RetrievedChunk(
                text="better duplicate",
                metadata={"source": "refund_policy.md", "chunk_index": 4},
                distance=0.3,
            ),
        ]

        deduped = dedupe_chunks(chunks)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].text, "better duplicate")

    def test_empty_query_result_returns_no_chunks(self) -> None:
        chunks = chunks_from_query_result({"documents": [[]], "metadatas": [[]]})

        self.assertEqual(chunks, [])
        self.assertEqual(answer_question("refund time?", chunks, DEFAULT_MODEL), NO_CONTEXT_MESSAGE)

    def test_context_and_answer_include_sources_and_chunk_text(self) -> None:
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
        self.assertIn(
            "Refunds are returned in 3-5 business days after approval.",
            format_answer("Refunds arrive in 3-5 business days.", chunks),
        )

    def test_format_answer_with_queries_prints_queries_before_answer(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Refunds are returned in 3-5 business days after approval.",
                metadata={"source": "refund_policy.md", "chunk_index": 3},
            )
        ]

        output = format_answer_with_queries(
            "\u7ed3\u8bba\uff1aRefunds arrive in 3-5 business days.",
            chunks,
            ["refund time?", "refund arrival days"],
        )

        self.assertLess(output.index("\u68c0\u7d22\u67e5\u8be2\uff1a"), output.index("\u7ed3\u8bba\uff1a"))
        self.assertIn("- refund time?", output)
        self.assertIn("- refund arrival days", output)
        self.assertIn("refund_policy.md#chunk=3", output)

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

    def test_parse_search_queries_accepts_json_and_lines(self) -> None:
        json_queries = parse_search_queries('["物流显示已送达但未收到", "运输途中丢失处理"]', 4)
        line_queries = parse_search_queries("1. 未收到商品怎么办\n- 联系客服补发退款", 4)

        self.assertEqual(json_queries, ["物流显示已送达但未收到", "运输途中丢失处理"])
        self.assertEqual(line_queries, ["未收到商品怎么办", "联系客服补发退款"])

    def test_generate_search_queries_keeps_original_question(self) -> None:
        with patch(
            "ai_knowledge_demo.ask.post_ollama_chat",
            return_value={"message": {"content": '["运输途中丢失处理", "联系客服补发退款"]'}},
        ):
            queries = generate_search_queries(
                "app shows delivered but not received",
                "qwen2.5:7b",
            )

        self.assertEqual(queries[0], "app shows delivered but not received")
        self.assertIn("运输途中丢失处理", queries)
        self.assertIn("联系客服补发退款", queries)

    def test_generate_search_queries_falls_back_to_original_on_error(self) -> None:
        with patch(
            "ai_knowledge_demo.ask.post_ollama_chat",
            side_effect=RuntimeError("Ollama failed"),
        ):
            queries = generate_search_queries("where is my order?", "qwen2.5:7b")

        self.assertEqual(queries, ["where is my order?"])

    def test_ollama_prompt_uses_three_level_answerability_template(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Electronic invoices are generated within 1 hour after payment.",
                metadata={"source": "refund_policy.md", "chunk_index": 7},
            )
        ]

        payload = build_ollama_payload("personal invoice?", chunks, "qwen2.5:7b")
        system_prompt = payload["messages"][0]["content"]

        self.assertIn("\u53ef\u76f4\u63a5\u56de\u7b54", system_prompt)
        self.assertIn("\u90e8\u5206\u76f8\u5173\u4f46\u672a\u660e\u786e", system_prompt)
        self.assertIn("\u5b8c\u5168\u65e0\u5173", system_prompt)
        self.assertIn("\u7ed3\u8bba\uff1a", system_prompt)
        self.assertIn("\u53ef\u4ee5\u786e\u8ba4\uff1a", system_prompt)
        self.assertIn("\u672a\u660e\u786e\u8bf4\u660e\uff1a", system_prompt)
        self.assertIn("\u5171\u4eab\u540c\u4e00\u4e1a\u52a1\u4e3b\u9898", system_prompt)
        self.assertIn("\u4efb\u4f55\u4e00\u6761", system_prompt)
        self.assertIn("\u4e0d\u8981\u8f93\u51fa\u6a21\u677f\u5916", system_prompt)
        self.assertIn(NO_CONTEXT_MESSAGE, system_prompt)

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

        self.assertIn("Refunds arrive in 3-5 business days.", answer)
        self.assertIn("\u7ed3\u8bba\uff1a", answer)
        self.assertIn("\u53ef\u4ee5\u786e\u8ba4\uff1a", answer)
        self.assertIn("\u672a\u660e\u786e\u8bf4\u660e\uff1a", answer)
        post_ollama_chat.assert_called_once()

    def test_normalize_answer_template_fixes_false_no_context(self) -> None:
        chunks = [
            RetrievedChunk(
                text="## Invoice\n\nElectronic invoices are generated after payment.",
                metadata={"source": "refund_policy.md", "chunk_index": 7},
            )
        ]
        answer = (
            f"\u7ed3\u8bba\uff1a{NO_CONTEXT_MESSAGE}\n"
            "\u53ef\u4ee5\u786e\u8ba4\uff1a\n"
            "- \u65e0\n"
            "\u672a\u660e\u786e\u8bf4\u660e\uff1a\n"
            "- Personal invoices are not mentioned."
        )

        normalized = normalize_answer_template(answer, "invoice?", chunks)

        self.assertNotIn(f"\u7ed3\u8bba\uff1a{NO_CONTEXT_MESSAGE}", normalized)
        self.assertIn("\u77e5\u8bc6\u5e93\u4e2d\u6709\u76f8\u5173\u4fe1\u606f", normalized)
        self.assertIn("\u53ef\u4ee5\u786e\u8ba4\uff1a", normalized)
        self.assertIn("\u672a\u660e\u786e\u8bf4\u660e\uff1a", normalized)

    def test_rerank_promotes_keyword_match_over_vector_distance(self) -> None:
        chunks = [
            RetrievedChunk(
                text="## 七、发票相关说明\n\n若订单已退款，对应发票将自动作废。",
                metadata={"source": "refund_policy.md", "chunk_index": 7},
                distance=0.7,
            ),
            RetrievedChunk(
                text="## 十、特殊商品说明\n\n电子类商品激活后，可能无法进行退款。",
                metadata={"source": "refund_policy.md", "chunk_index": 10},
                distance=0.8,
            ),
            RetrievedChunk(
                text="## 三、不可退款情况\n\n超过退款申请时效可能无法退款。",
                metadata={"source": "refund_policy.md", "chunk_index": 3},
                distance=0.9,
            ),
            RetrievedChunk(
                text="## 五、优惠券退款规则\n\n退款后优惠券通常无法返还。",
                metadata={"source": "refund_policy.md", "chunk_index": 5},
                distance=0.95,
            ),
            RetrievedChunk(
                text=(
                    "## 二、退款到账时间\n\n"
                    "审核通过后，款项将在 3-5 个工作日内原路返回。\n\n"
                    "部分国际银行卡退款时间可能延长至 7-15 个工作日。"
                ),
                metadata={"source": "refund_policy.md", "chunk_index": 2},
                distance=1.02,
            ),
        ]

        reranked = rerank_chunks("退款多久到账？", chunks, top_k=4)

        self.assertEqual(len(reranked), 4)
        self.assertEqual(reranked[0].metadata["chunk_index"], 2)

    def test_rerank_empty_candidates_stays_empty(self) -> None:
        self.assertEqual(rerank_chunks("退款多久到账？", [], top_k=4), [])


    def test_rerank_uses_generated_search_queries_for_scoring(self) -> None:
        chunks = [
            RetrievedChunk(
                text="## Special items\n\nActivated electronics may not be refundable.",
                metadata={"source": "refund_policy.md", "chunk_index": 10},
                distance=0.7,
            ),
            RetrievedChunk(
                text=(
                    "## \u7269\u6d41\u76f8\u5173\u95ee\u9898\n\n"
                    "\u82e5\u5546\u54c1\u5728\u8fd0\u8f93\u8fc7\u7a0b\u4e2d\u4e22\u5931\uff0c"
                    "\u7528\u6237\u53ef\u8054\u7cfb\u5ba2\u670d\u91cd\u65b0\u8865\u53d1"
                    "\u6216\u7533\u8bf7\u5168\u989d\u9000\u6b3e\u3002"
                ),
                metadata={"source": "refund_policy.md", "chunk_index": 4},
                distance=1.1,
            ),
        ]

        reranked = rerank_chunks(
            "app shows delivered but not received",
            chunks,
            top_k=1,
            scoring_queries=[
                "app shows delivered but not received",
                "\u7269\u6d41\u5f02\u5e38 \u8fd0\u8f93\u4e22\u5931 "
                "\u672a\u6536\u8d27 \u5ba2\u670d \u8865\u53d1 \u5168\u989d\u9000\u6b3e",
            ],
        )

        self.assertEqual(reranked[0].metadata["chunk_index"], 4)


if __name__ == "__main__":
    unittest.main()
