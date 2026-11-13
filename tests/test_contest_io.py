import json
from pathlib import Path
import tempfile
import unittest

from submission.task_io import load_tasks, make_prompt, save_answer, apply_code_answer


class TaskIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task_dir = self.root / "example"
        self.workspace = self.task_dir / "workspace"
        self.workspace.mkdir(parents=True)
        self.raw = {"task_id": "example", "task_type": "math", "instruction": "Calculate 8*9",
                    "reward": 5, "check": "exact_int", "expected": "HIDDEN_CANARY_781234"}
        (self.task_dir / "task.json").write_text(json.dumps(self.raw))

    def task(self):
        return load_tasks(self.root)[0]

    def test_expected_not_exposed_and_nested_metadata_not_sent(self):
        (self.workspace / "task.json").write_text(json.dumps(self.raw))
        (self.workspace / "answer.txt").write_text("HIDDEN_CANARY_781234")
        task = self.task()
        self.assertNotIn("expected", task)
        self.assertNotIn("HIDDEN_CANARY", make_prompt(task))

    def test_answer_beside_task_json_and_format_validation(self):
        task = self.task()
        for value in (None, "", "72 because", "72\n73"):
            self.assertFalse(save_answer(task, value))
        self.assertTrue(save_answer(task, "72"))
        self.assertEqual((self.task_dir / "answer.txt").read_text(), "72\n")
        self.assertFalse((self.workspace / "answer.txt").exists())

    def test_large_snapshot_not_silently_truncated(self):
        (self.workspace / "data.csv").write_text("x" * 200)
        self.assertIsNone(make_prompt(self.task(), max_bytes=100))

    def test_model_patch_cannot_edit_tests_or_escape(self):
        (self.workspace / "mylib.py").write_text("def f(): return 1\n")
        test = "import unittest\nfrom mylib import f\nclass T(unittest.TestCase):\n def test_f(self): self.assertEqual(f(),2)\n"
        (self.workspace / "test_mylib.py").write_text(test)
        task = dict(self.task(), task_type="coding", check="unittest")
        for name in ("test_mylib.py", "../escape.py", "/tmp/escape.py", "missing.py"):
            self.assertFalse(apply_code_answer(task, json.dumps({"files": {name: "pass"}})))
        self.assertFalse(apply_code_answer(task, json.dumps({"files": {"mylib.py": "def f(): return 3"}})))
        self.assertEqual((self.workspace / "mylib.py").read_text(), "def f(): return 1\n")
        self.assertTrue(apply_code_answer(task, json.dumps({"files": {"mylib.py": "def f(): return 2"}})))
        self.assertEqual((self.workspace / "test_mylib.py").read_text(), test)

    def test_duplicate_ids_rejected(self):
        other = self.root / "other"
        other.mkdir()
        (other / "task.json").write_text(json.dumps(self.raw))
        with self.assertRaises(ValueError):
            load_tasks(self.root)


if __name__ == "__main__":
    unittest.main()
