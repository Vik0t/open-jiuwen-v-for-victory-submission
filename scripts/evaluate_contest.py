"""Offline example evaluator. Expected answers belong only to this process."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(workspace):
    hashes = {}
    if workspace.is_symlink():
        raise ValueError("workspace symlink is not allowed")
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink():
            raise ValueError("workspace symlinks are not allowed")
        if path.is_file() and "__pycache__" not in path.parts:
            hashes[str(path.relative_to(workspace))] = _sha(path)
    return hashes


def _unittest(workspace, timeout):
    harness = ("import json, sys, unittest\nsys.path.insert(0, '.')\n"
               "s = unittest.defaultTestLoader.discover('.', pattern='test*.py')\n"
               "r = unittest.TextTestRunner(verbosity=0).run(s)\n"
               "print('CONTEST_TEST_RESULT=' + json.dumps(dict(tests=r.testsRun, "
               "failures=len(r.failures), errors=len(r.errors), skipped=len(r.skipped), "
               "expected_failures=len(r.expectedFailures), "
               "passed=r.wasSuccessful())))\n"
               "sys.exit(0 if r.wasSuccessful() else 1)\n")
    with tempfile.TemporaryDirectory(prefix="contest-eval-") as tmp:
        shutil.copytree(workspace, Path(tmp) / "workspace", ignore=shutil.ignore_patterns("__pycache__"))
        result = subprocess.run([sys.executable, "-I", "-B", "-c", harness],
                                cwd=Path(tmp) / "workspace", timeout=timeout,
                                env={"PATH": os.defpath}, capture_output=True, text=True)
    lines = [line.split("=", 1)[1] for line in result.stdout.splitlines()
             if line.startswith("CONTEST_TEST_RESULT=")]
    if len(lines) != 1:
        return {"tests": 0, "passed": False, "reason": "test process failed"}
    summary = json.loads(lines[0])
    summary["passed"] = (result.returncode == 0 and summary["passed"] and summary["tests"] > 0
                         and summary["errors"] == 0 and summary["failures"] == 0
                         and summary["skipped"] == 0 and summary["expected_failures"] == 0)
    return summary


def evaluate(tasks_dir, reference_dir=None, timeout=10):
    """Score answer files and fresh public tests; never mutate task workspaces."""
    if not 0 < timeout <= 10:
        raise ValueError("timeout must be greater than zero and at most 10 seconds")
    root = Path(tasks_dir).resolve(strict=True)
    reference = Path(reference_dir).resolve(strict=True) if reference_dir else None
    source = reference or root
    results = []
    for metadata in sorted(source.glob("*/task.json")):
        if not metadata.resolve().is_relative_to(source):
            raise ValueError("task metadata escapes tasks directory")
        task = json.loads(metadata.read_text())
        reward = task["reward"]
        if type(reward) not in (int, float) or reward < 0 or not math.isfinite(reward):
            raise ValueError("reward must be finite and nonnegative")
        folder = root / metadata.parent.relative_to(source)
        row = {"task_id": task["task_id"], "check": task["check"], "reward": reward,
               "passed": False, "status": "not_checked", "source_hashes": {}, "test_hashes": {},
               "test_integrity_verified": False, "reference_task_sha256": _sha(metadata)}
        results.append(row)
        try:
            candidate = folder / "task.json"
            if not candidate.resolve().is_relative_to(root):
                raise ValueError("candidate task path escapes tasks directory")
            row["task_sha256"] = _sha(candidate)
            if reference and row["task_sha256"] != row["reference_task_sha256"]:
                row["status"] = "task_metadata_changed"
                continue
            row["source_hashes"] = _hashes(folder / "workspace")
            row["test_hashes"] = {p: h for p, h in row["source_hashes"].items()
                                  if Path(p).name.startswith("test") and p.endswith(".py")}
            if reference and task["check"] == "unittest":
                original = _hashes(metadata.parent / "workspace")
                original = {p: h for p, h in original.items()
                            if Path(p).name.startswith("test") and p.endswith(".py")}
                row["reference_test_hashes"] = original
                if not original or original != row["test_hashes"]:
                    row["status"] = "public_tests_changed_or_missing"
                    continue
                row["test_integrity_verified"] = True
            answer_file = folder / "answer.txt"
            if answer_file.is_symlink():
                raise ValueError("answer symlink is not allowed")
            answer = answer_file.read_text(encoding="utf-8").strip()
            row["answer_sha256"] = _sha(answer_file)
            check, expected = task["check"], task.get("expected")
            if check == "unittest":
                row["tests"] = _unittest(folder / "workspace", timeout) if answer == "DONE" else {}
                row["passed"] = answer == "DONE" and row["tests"].get("passed", False)
            elif expected is None:
                row["status"] = "expected_unavailable"
                continue
            elif check == "exact_int":
                expected = str(expected).strip()
                row["passed"] = (bool(re.fullmatch(r"[+-]?\d+", answer))
                                 and bool(re.fullmatch(r"[+-]?\d+", expected))
                                 and int(answer) == int(expected))
            elif check == "exact_str":
                row["passed"] = isinstance(expected, str) and answer == expected.strip()
            else:
                row["status"] = "unsupported_check"
                continue
            row["status"] = "passed" if row["passed"] else "failed"
        except subprocess.TimeoutExpired:
            row["status"] = "test_timeout"
        except (OSError, UnicodeError, ValueError):
            row["status"] = "missing_or_invalid_artifact"
    return {"total_reward": sum(r["reward"] for r in results if r["passed"]),
            "total_possible": sum(r["reward"] for r in results), "count_passed": sum(r["passed"] for r in results),
            "count_tasks": len(results), "reference_dir": str(reference) if reference else None, "by_task": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    report = evaluate(args.tasks_dir, args.reference_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
