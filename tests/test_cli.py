import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ai_knowledge_demo.cli import main


class CliTests(unittest.TestCase):
    def test_main_prints_ready_message_and_returns_success(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), "ai-knowledge-demo is ready.\n")


if __name__ == "__main__":
    unittest.main()
