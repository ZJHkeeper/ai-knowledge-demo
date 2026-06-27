import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ai_knowledge_demo.ask import NO_CONTEXT_MESSAGE, RetrievedChunk
from ai_knowledge_demo.evaluate import (
    DEFAULT_CASES_PATH,
    EvalResult,
    check_answer,
    check_retrieval,
    check_should_include,
    evaluate_case,
    load_cases,
    main,
    validate_case,
)


class EvaluateTests(unittest.TestCase):
    def test_eval_cases_json_loads_and_is_valid(self) -> None:
        cases = load_cases(DEFAULT_CASES_PATH)

        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual([error for case in cases for error in validate_case(case)], [])
        self.assertIn("type", cases[0])
        self.assertIn("expected_sources", cases[0])
        self.assertIn("must_include", cases[0])
        self.assertIn("expected_refusal", cases[0])

    def test_check_retrieval_matches_any_expected_source(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Refunds arrive in 3-5 business days.",
                metadata={"source": "refund_policy.md", "chunk_index": 2},
            ),
        ]

        failures = check_retrieval(
            chunks,
            ["refund_policy.md", "customer_service_manual.md"],
        )

        self.assertEqual(failures, [])

    def test_check_retrieval_reports_missing_source(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Invoices are generated after payment.",
                metadata={"source": "invoice.md", "chunk_index": 1},
            )
        ]

        failures = check_retrieval(chunks, ["refund_policy.md"])

        self.assertEqual(len(failures), 1)
        self.assertIn("retrieval missing expected source", failures[0])
        self.assertIn("refund_policy.md", failures[0])

    def test_check_answer_reports_required_terms_and_unexpected_refusal(self) -> None:
        failures = check_answer(
            NO_CONTEXT_MESSAGE,
            must_include=["3-5"],
            expected_refusal=False,
        )

        self.assertIn("answer missing required term=3-5", failures)
        self.assertIn("answer unexpectedly refused", failures)

    def test_check_answer_allows_expected_refusal(self) -> None:
        failures = check_answer(
            NO_CONTEXT_MESSAGE,
            must_include=[NO_CONTEXT_MESSAGE],
            expected_refusal=True,
        )

        self.assertEqual(failures, [])

    def test_check_should_include_returns_warnings_only(self) -> None:
        warnings = check_should_include(
            "Conclusion: the knowledge base has invoice-related information.",
            ["not explicitly stated", "personal invoice"],
        )

        self.assertEqual(
            warnings,
            [
                "answer missing suggested term=not explicitly stated",
                "answer missing suggested term=personal invoice",
            ],
        )

    def test_evaluate_case_retrieval_only_still_runs_query_rewrite(self) -> None:
        case = {
            "id": "EVAL-001",
            "type": "basic_fact",
            "type_description": "Basic fact question.",
            "question": "refund time?",
            "expected_sources": ["refund_policy.md"],
            "must_include": ["3-5"],
            "expected_refusal": False,
        }
        chunks = [
            RetrievedChunk(
                text="Refunds arrive in 3-5 business days.",
                metadata={"source": "refund_policy.md", "chunk_index": 2},
            )
        ]

        with (
            patch("ai_knowledge_demo.evaluate.generate_search_queries", return_value=["refund time?", "refund arrival"]) as generate,
            patch("ai_knowledge_demo.evaluate.answer_question") as answer,
            patch("ai_knowledge_demo.evaluate.retrieve_chunks", return_value=chunks) as retrieve,
        ):
            result = evaluate_case(
                case,
                persist_dir=PROJECT_ROOT / "chroma_db",
                collection_name="ai_knowledge_demo",
                top_k=4,
                bm25_index_path=None,
                use_bm25=True,
                model="qwen2.5:7b",
                ollama_url="http://localhost:11434",
                retrieval_only=True,
                use_query_rewrite=True,
            )

        self.assertTrue(result.passed)
        self.assertIsNone(result.answer)
        self.assertEqual(result.warnings, [])
        generate.assert_called_once_with("refund time?", "qwen2.5:7b", "http://localhost:11434")
        answer.assert_not_called()
        self.assertEqual(retrieve.call_args.kwargs["search_queries"], ["refund time?", "refund arrival"])

    def test_evaluate_case_can_disable_query_rewrite(self) -> None:
        case = {
            "id": "EVAL-001",
            "type": "basic_fact",
            "type_description": "Basic fact question.",
            "question": "refund time?",
            "expected_sources": ["refund_policy.md"],
            "must_include": ["3-5"],
            "expected_refusal": False,
        }
        chunks = [
            RetrievedChunk(
                text="Refunds arrive in 3-5 business days.",
                metadata={"source": "refund_policy.md", "chunk_index": 2},
            )
        ]

        with (
            patch("ai_knowledge_demo.evaluate.generate_search_queries") as generate,
            patch("ai_knowledge_demo.evaluate.retrieve_chunks", return_value=chunks) as retrieve,
        ):
            result = evaluate_case(
                case,
                persist_dir=PROJECT_ROOT / "chroma_db",
                collection_name="ai_knowledge_demo",
                top_k=4,
                bm25_index_path=None,
                use_bm25=True,
                model="qwen2.5:7b",
                ollama_url="http://localhost:11434",
                retrieval_only=True,
                use_query_rewrite=False,
            )

        self.assertTrue(result.passed)
        generate.assert_not_called()
        self.assertEqual(retrieve.call_args.kwargs["search_queries"], ["refund time?"])

    def test_evaluate_case_checks_generated_answer(self) -> None:
        case = {
            "id": "EVAL-001",
            "type": "basic_fact",
            "type_description": "Basic fact question.",
            "question": "refund time?",
            "expected_sources": ["refund_policy.md"],
            "must_include": ["3-5", "business days"],
            "should_include": ["original payment method"],
            "expected_refusal": False,
        }
        chunks = [
            RetrievedChunk(
                text="Refunds arrive in 3-5 business days.",
                metadata={"source": "refund_policy.md", "chunk_index": 2},
            )
        ]

        with (
            patch("ai_knowledge_demo.evaluate.generate_search_queries", return_value=["refund time?"]),
            patch("ai_knowledge_demo.evaluate.retrieve_chunks", return_value=chunks),
            patch(
                "ai_knowledge_demo.evaluate.answer_question",
                return_value="Conclusion: refunds arrive in 3-5 business days.",
            ),
        ):
            result = evaluate_case(
                case,
                persist_dir=PROJECT_ROOT / "chroma_db",
                collection_name="ai_knowledge_demo",
                top_k=4,
                bm25_index_path=None,
                use_bm25=True,
                model="qwen2.5:7b",
                ollama_url="http://localhost:11434",
                retrieval_only=False,
                use_query_rewrite=True,
            )

        self.assertTrue(result.passed)
        self.assertIn("3-5", result.answer or "")
        self.assertEqual(
            result.warnings,
            ["answer missing suggested term=original payment method"],
        )

    def test_main_returns_success_when_all_cases_pass(self) -> None:
        output = io.StringIO()
        with (
            patch("ai_knowledge_demo.evaluate.load_cases", return_value=[]),
            patch(
                "ai_knowledge_demo.evaluate.evaluate_cases",
                return_value=[EvalResult("ok", True, [], ["refund_policy.md#chunk=2"])],
            ),
            redirect_stdout(output),
        ):
            exit_code = main(["--retrieval-only"])

        self.assertEqual(exit_code, 0)
        self.assertIn("[PASS] ok", output.getvalue())
        self.assertIn("Summary: 1/1 passed", output.getvalue())

    def test_main_returns_failure_when_any_case_fails(self) -> None:
        output = io.StringIO()
        with (
            patch("ai_knowledge_demo.evaluate.load_cases", return_value=[]),
            patch(
                "ai_knowledge_demo.evaluate.evaluate_cases",
                return_value=[EvalResult("bad", False, ["answer missing required term=3-5"], [])],
            ),
            redirect_stdout(output),
        ):
            exit_code = main(["--retrieval-only"])

        self.assertEqual(exit_code, 1)
        self.assertIn("[FAIL] bad", output.getvalue())

    def test_main_reports_ollama_error_clearly(self) -> None:
        error_output = io.StringIO()
        with (
            patch("ai_knowledge_demo.evaluate.load_cases", return_value=[]),
            patch(
                "ai_knowledge_demo.evaluate.evaluate_cases",
                side_effect=RuntimeError("Cannot connect to Ollama."),
            ),
            redirect_stderr(error_output),
        ):
            exit_code = main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("Evaluation failed: Cannot connect to Ollama.", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
