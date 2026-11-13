import json
from pathlib import Path
import shutil

import pytest

from scripts.evaluate_contest import evaluate


def make_task(root, name, check, expected, answer=None, test=None, reward=3):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "task.json").write_text(json.dumps(
        {"task_id": name, "check": check, "expected": expected, "reward": reward}))
    workspace = folder / "workspace"
    workspace.mkdir()
    if answer is not None:
        (folder / "answer.txt").write_text(answer)
    if test is not None:
        (workspace / "test_public.py").write_text(test)
    return folder


def test_integer_and_string_rules_and_missing_answer(tmp_path):
    make_task(tmp_path, "a", "exact_int", "37", " +0037\n", reward=5)
    make_task(tmp_path, "b", "exact_str", "Blue  House", "\nBlue  House \n", reward=2)
    make_task(tmp_path, "c", "exact_str", "Blue House", "blue House", reward=7)
    make_task(tmp_path, "d", "exact_int", "37", "37.0", reward=11)
    make_task(tmp_path, "e", "exact_int", "37", reward=13)
    result = evaluate(tmp_path)
    assert (result["total_reward"], result["total_possible"], result["count_passed"]) == (7, 38, 2)
    assert result["by_task"][-1]["status"] == "missing_or_invalid_artifact"
    assert "Blue  House" not in json.dumps(result)


PASS = "import unittest\nclass TestPublic(unittest.TestCase):\n def test_ok(self): self.assertEqual(2+3,5)\n"


def test_coding_requires_done_and_nonzero_passing_tests(tmp_path):
    make_task(tmp_path, "a", "unittest", "DONE", "DONE", PASS)
    make_task(tmp_path, "b", "unittest", "DONE", "finished", PASS)
    make_task(tmp_path, "c", "unittest", "DONE", "DONE", "# no collected test\n")
    make_task(tmp_path, "d", "unittest", "DONE", "DONE", "raise ImportError('bad')\n")
    result = evaluate(tmp_path)
    assert result["count_passed"] == 1
    assert result["by_task"][0]["tests"]["tests"] == 1
    assert result["by_task"][2]["tests"]["tests"] == 0
    assert result["by_task"][3]["tests"]["errors"] == 1
    assert not list(tmp_path.rglob("__pycache__"))


def test_reference_rejects_edited_tests_and_metadata(tmp_path):
    reference = tmp_path / "reference"
    make_task(reference, "a", "unittest", "DONE", "DONE", PASS)
    candidate = tmp_path / "candidate"
    shutil.copytree(reference, candidate)
    assert evaluate(candidate, reference)["count_passed"] == 1
    (candidate / "a/workspace/test_public.py").write_text(PASS + "# altered\n")
    result = evaluate(candidate, reference)
    assert result["by_task"][0]["status"] == "public_tests_changed_or_missing"
    shutil.copyfile(reference / "a/workspace/test_public.py", candidate / "a/workspace/test_public.py")
    metadata = candidate / "a/task.json"
    task = json.loads(metadata.read_text())
    task["reward"] = 999
    metadata.write_text(json.dumps(task))
    result = evaluate(candidate, reference)
    assert result["by_task"][0]["status"] == "task_metadata_changed"
    assert result["total_possible"] == 3 and result["total_reward"] == 0


def test_public_test_imports_local_candidate_module_in_isolated_python(tmp_path):
    folder = make_task(tmp_path, "a", "unittest", "DONE", "DONE",
        "import unittest\nfrom mylib import increment\n"
        "class TestPublic(unittest.TestCase):\n"
        " def test_increment(self): self.assertEqual(increment(26),27)\n")
    (folder / "workspace/mylib.py").write_text("def increment(n): return n+1\n")
    result = evaluate(tmp_path)
    assert result["count_passed"] == 1
    assert result["by_task"][0]["tests"]["tests"] == 1
    assert "mylib.py" in result["by_task"][0]["source_hashes"]
    assert not list(tmp_path.rglob("__pycache__"))


def test_timeout_and_skipped_tests_never_pass(tmp_path):
    make_task(tmp_path, "a", "unittest", "DONE", "DONE", "import time\ntime.sleep(1)\n" + PASS)
    result = evaluate(tmp_path, timeout=0.05)
    assert result["by_task"][0]["status"] == "test_timeout"
    with pytest.raises(ValueError):
        evaluate(tmp_path, timeout=11)
    (tmp_path / "a/workspace/test_public.py").write_text(
        "import unittest\nclass TestPublic(unittest.TestCase):\n"
        " @unittest.skip('not executed')\n def test_skip(self): pass\n")
    result = evaluate(tmp_path)
    assert not result["by_task"][0]["passed"]
    assert result["by_task"][0]["tests"]["skipped"] == 1


def test_workspace_and_answer_symlinks_are_rejected(tmp_path):
    root = tmp_path / "tasks"
    folder = make_task(root, "a", "exact_int", "19", "19")
    outside = tmp_path / "outside.txt"
    outside.write_text("19")
    (folder / "workspace/link.txt").symlink_to(outside)
    assert evaluate(root)["count_passed"] == 0
    (folder / "workspace/link.txt").unlink()
    (folder / "answer.txt").unlink()
    (folder / "answer.txt").symlink_to(outside)
    assert evaluate(root)["count_passed"] == 0


def test_late_fake_summary_cannot_override_real_test_failure(tmp_path):
    fake = json.dumps({"tests": 1, "failures": 0, "errors": 0, "skipped": 0, "passed": True})
    test = ("import atexit, unittest\n"
            f"atexit.register(lambda: print('CONTEST_TEST_RESULT=' + {fake!r}))\n"
            "class TestPublic(unittest.TestCase):\n def test_bad(self): self.fail('real failure')\n")
    make_task(tmp_path, "a", "unittest", "DONE", "DONE", test)
    result = evaluate(tmp_path)
    assert result["count_passed"] == 0
    assert not result["by_task"][0]["tests"]["passed"]


def test_expected_failure_is_not_an_executed_passing_assertion(tmp_path):
    make_task(tmp_path, "a", "unittest", "DONE", "DONE",
        "import unittest\nclass T(unittest.TestCase):\n"
        " @unittest.expectedFailure\n def test_bad(self): self.fail('not a pass')\n")
    result = evaluate(tmp_path)
    assert result["count_passed"] == 0
    assert result["by_task"][0]["tests"]["expected_failures"] == 1


def test_cli_writes_json_without_expected_answer_contents(tmp_path):
    from scripts.evaluate_contest import main
    from unittest.mock import patch
    root = tmp_path / "tasks"
    make_task(root, "a", "exact_str", "synthetic-hidden-answer", "different")
    output = tmp_path / "report.json"
    with patch("sys.argv", ["evaluate_contest.py", "--tasks-dir", str(root), "--output", str(output)]):
        main()
    assert json.loads(output.read_text())["count_tasks"] == 1
    assert "synthetic-hidden-answer" not in output.read_text()
