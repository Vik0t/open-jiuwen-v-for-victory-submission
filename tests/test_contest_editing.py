from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from submission.editing import MAX_SOURCE_BYTES, apply_edit, write_source


class SourceEditingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source = self.workspace / "solver.py"
        self.original = b"# Original comment\nvalue = 1\n"
        self.source.write_bytes(self.original)

    def test_exact_edit_preserves_utf8_bom_line_endings_comments_and_mode(self):
        original = "\ufeff# Пример\r\nvalue = 1\r\n# Ничего не удалять\r\n".encode("utf-8")
        self.source.write_bytes(original)
        self.source.chmod(0o640)
        result = apply_edit(self.workspace, "solver.py", "value = 1", "value = 2")
        self.assertTrue(result["success"], result)
        self.assertEqual(self.source.read_bytes(), original.replace(b"value = 1", b"value = 2"))
        self.assertEqual(stat.S_IMODE(self.source.stat().st_mode), 0o640)

    def test_missing_empty_duplicate_and_overlapping_fragments_leave_file_unchanged(self):
        self.source.write_text('value = "aaa"\n', encoding="utf-8")
        original = self.source.read_bytes()
        for before in ("", "missing", "a", "aa"):
            with self.subTest(before=before):
                result = apply_edit(self.workspace, "solver.py", before, "b")
                self.assertFalse(result["success"], result)
                self.assertFalse(result["changed"])
                self.assertEqual(self.source.read_bytes(), original)

    def test_paths_cannot_escape_or_overwrite_tests_and_metadata(self):
        outside = self.root / "outside.py"
        outside.write_bytes(self.original)
        names = ("../outside.py", str(outside), "..\\outside.py", "test_solver.py",
                 "solver_test.py", "conftest.py", "tests/helper.py", "Test/helper.py",
                 "answers/solver.py", "metadata/task.py", ".git/config.py",
                 "task.json", "answer.txt", ".", "", "solver\n.py")
        for name in names:
            with self.subTest(path=name):
                result = write_source(self.workspace, name, "value = 2\n")
                self.assertFalse(result["success"], result)
        self.assertEqual(outside.read_bytes(), self.original)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(list(self.workspace.iterdir()), [self.source])

    def test_file_parent_dangling_and_workspace_symlinks_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "solver.py").write_bytes(self.original)
        (self.workspace / "linked.py").symlink_to(outside / "solver.py")
        (self.workspace / "linked_dir").symlink_to(outside, target_is_directory=True)
        (self.workspace / "dangling.py").symlink_to(outside / "missing.py")
        for name in ("linked.py", "linked_dir/solver.py", "dangling.py"):
            result = write_source(self.workspace, name, "value = 2\n")
            self.assertFalse(result["success"], result)
        alias = self.root / "alias"
        alias.symlink_to(self.workspace, target_is_directory=True)
        self.assertFalse(write_source(alias, "solver.py", "value = 2\n")["success"])
        self.assertEqual((outside / "solver.py").read_bytes(), self.original)

    def test_byte_limit_counts_utf8_and_failed_edits_leave_source_unchanged(self):
        text = "#" + "я" * (MAX_SOURCE_BYTES // 2) + "\n"
        self.assertLess(len(text), MAX_SOURCE_BYTES)
        result = write_source(self.workspace, "solver.py", text)
        self.assertFalse(result["success"], result)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.source.write_bytes(b"#" * (MAX_SOURCE_BYTES + 1))
        result = apply_edit(self.workspace, "solver.py", "#", "")
        self.assertFalse(result["success"], result)
        self.assertEqual(self.source.stat().st_size, MAX_SOURCE_BYTES + 1)

    def test_invalid_python_and_markdown_are_rejected_without_silent_cleanup(self):
        for text in ("def broken(\n", "```python\nvalue = 2\n```\n"):
            result = write_source(self.workspace, "solver.py", text)
            self.assertFalse(result["success"], result)
            self.assertIn("Invalid Python", result["error"])
            self.assertEqual(self.source.read_bytes(), self.original)
        result = apply_edit(self.workspace, "solver.py", "value = 1", "value = (")
        self.assertFalse(result["success"], result)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_new_nested_source_is_written_but_never_executed(self):
        marker = self.workspace / "marker"
        text = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        result = write_source(self.workspace, "helpers/analyze.py", text)
        self.assertTrue(result["success"], result)
        self.assertTrue(result["created"])
        self.assertEqual((self.workspace / "helpers/analyze.py").read_text(), text)
        self.assertFalse(marker.exists())

    def test_unchanged_edit_is_reported_without_replacing_file(self):
        with patch("submission.editing.os.replace") as replace:
            result = apply_edit(self.workspace, "solver.py", "value = 1", "value = 1")
        self.assertTrue(result["success"], result)
        self.assertFalse(result["changed"])
        replace.assert_not_called()

    def test_failed_atomic_replace_preserves_original_and_removes_temporary_file(self):
        with patch("submission.editing.os.replace", side_effect=OSError("simulated failure")):
            result = apply_edit(self.workspace, "solver.py", "value = 1", "value = 2")
        self.assertFalse(result["success"], result)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(list(self.workspace.iterdir()), [self.source])


if __name__ == "__main__":
    unittest.main()
