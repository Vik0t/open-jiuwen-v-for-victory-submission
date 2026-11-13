"""Small test-guided Python repairs using generic AST mutations, without an LLM.

Public tests guide the search. Passing them is evidence, not a hidden-test guarantee.
Only a fully passing candidate is copied back; tests are never edited.
"""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


TEST_RUNNER = """
import json, sys, unittest
sys.path.insert(0, '.')
suite = unittest.defaultTestLoader.discover('.', pattern='test*.py')
result = unittest.TextTestRunner(verbosity=0).run(suite)
print('__TEST_RESULT__' + json.dumps({
    'tests': result.testsRun,
    'failures': len(result.failures) + len(result.errors),
    'skipped': len(result.skipped),
    'expected_failures': len(result.expectedFailures),
    'unexpected': len(result.unexpectedSuccesses)}), flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def run_tests(workspace, timeout=3):
    """Run provided unittest files; an empty/failed discovery cannot count as success."""
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    try:
        run = subprocess.run([sys.executable, "-I", "-B", "-c", TEST_RUNNER],
                             cwd=workspace, env=environment, capture_output=True,
                             text=True, timeout=timeout)
        lines = [line for line in run.stdout.splitlines() if line.startswith("__TEST_RESULT__")]
        if len(lines) == 1:
            result = json.loads(lines[0].removeprefix("__TEST_RESULT__"))
            failed = result["failures"] > 0 or result["unexpected"] > 0
            if result["tests"] > 0 and run.returncode == int(failed):
                return result
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return None


def _passing(result):
    return (result is not None and result["failures"] == 0
            and result["skipped"] == 0 and result["unexpected"] == 0
            and result["expected_failures"] == 0)


def _mutations(tree):
    """Enumerate one syntactic change at a time; no function/task names are used."""
    arithmetic = {ast.Add: (ast.Sub, ast.Mult), ast.Sub: (ast.Add,),
                  ast.Mult: (ast.Add, ast.FloorDiv), ast.FloorDiv: (ast.Mult,)}
    comparisons = {ast.Lt: (ast.Gt, ast.LtE), ast.Gt: (ast.Lt, ast.GtE),
                   ast.LtE: (ast.GtE, ast.Lt), ast.GtE: (ast.LtE, ast.Gt),
                   ast.Eq: (ast.NotEq,), ast.NotEq: (ast.Eq,)}
    call_pairs = {"sorted": "reversed", "reversed": "sorted", "min": "max", "max": "min"}
    for index, node in enumerate(ast.walk(tree)):
        edits = []
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in call_pairs:
            new_function = copy.deepcopy(node.func)
            new_function.id = call_pairs[node.func.id]
            edits.append(("func", new_function))
        if isinstance(node, (ast.AugAssign, ast.BinOp)):
            edits.extend(("op", operator()) for operator in arithmetic.get(type(node.op), ()))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            edits.extend(("ops", [operator()]) for operator in comparisons.get(type(node.ops[0]), ()))
            if len(node.comparators) == 1:
                value = node.comparators[0]
                if isinstance(value, ast.Constant) and type(value.value) is int and value.value in (0, 1):
                    edits.append(("comparators", [ast.Constant(1 - value.value)]))
        for field, value in edits:
            candidate = copy.deepcopy(tree)
            changed = list(ast.walk(candidate))[index]
            setattr(changed, field, value)
            yield ast.unparse(ast.fix_missing_locations(candidate)) + "\n"


def solve_code(instruction, workspace: Path, max_attempts=100, seconds=25):
    """Greedily reduce public test failures, then commit only a fully passing repair."""
    root = Path(workspace)
    result = {"status": "unsupported", "answer": None, "reason": "no_verified_repair", "attempts": 0}
    if not isinstance(instruction, str) or "test" not in instruction.casefold():
        return result
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        return dict(result, status="unknown", reason="linked_workspace_file")
    sources = [p.relative_to(root) for p in paths if p.is_file() and p.suffix == ".py"
               and not p.name.startswith("test") and p.name != "conftest.py"]
    if not sources:
        return result
    deadline = time.monotonic() + seconds
    with tempfile.TemporaryDirectory(prefix="contest_repair_") as directory:
        scratch = Path(directory) / "workspace"
        shutil.copytree(root, scratch, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "answer.txt"))
        baseline = run_tests(scratch, min(3, max(.01, seconds)))
        if baseline is None:
            return dict(result, status="unknown", reason="tests_unavailable")
        result["tests"] = baseline["tests"]
        changed = set()
        while not _passing(baseline) and result["attempts"] < max_attempts and time.monotonic() < deadline:
            improved = False
            for relative in sorted(sources):
                path = scratch / relative
                original = path.read_text(encoding="utf-8")
                try:
                    candidates = _mutations(ast.parse(original))
                    for candidate in candidates:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or result["attempts"] >= max_attempts:
                            break
                        path.write_text(candidate, encoding="utf-8")
                        measured = run_tests(scratch, min(3, remaining))
                        result["attempts"] += 1
                        if (measured is not None and measured["tests"] == baseline["tests"]
                                and measured["failures"] < baseline["failures"]
                                and measured["skipped"] == baseline["skipped"] == 0
                                and measured["expected_failures"] == baseline["expected_failures"] == 0
                                and measured["unexpected"] == 0):
                            baseline, improved = measured, True
                            changed.add(relative)
                            break
                        path.write_text(original, encoding="utf-8")
                except (SyntaxError, UnicodeError):
                    continue
                if improved:
                    break
            if not improved:
                break
        if not _passing(baseline):
            return result
        for relative in sorted(changed):
            target = root / relative
            temporary = target.with_name(target.name + ".repair-tmp")
            temporary.write_bytes((scratch / relative).read_bytes())
            temporary.replace(target)
        return dict(result, status="solved", answer="DONE", reason="public_tests_pass",
                    changed_files=[str(path) for path in sorted(changed)])
