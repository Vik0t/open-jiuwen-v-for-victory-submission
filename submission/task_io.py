"""Contest input/output. Hidden expected answers never enter solver task objects."""
import json
import os
from pathlib import Path
import re
import tempfile


def load_tasks(directory):
    tasks = []
    for path in sorted(Path(directory).glob("*/task.json")):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Linked task directory")
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Explicit projection: do not pass the raw metadata to any solver/model.
        task = {key: raw[key] for key in ("task_id", "task_type", "instruction", "reward", "check")}
        if (not all(isinstance(task[key], str) for key in ("task_id", "task_type", "instruction", "check"))
                or type(task["reward"]) is not int or task["reward"] < 0):
            raise ValueError("Invalid public task metadata")
        task.update(directory=path.parent.resolve(), workspace=(path.parent / "workspace").resolve())
        if (path.parent / "workspace").is_symlink():
            raise ValueError("Linked workspace")
        tasks.append(task)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task ID")
    return tasks


def atomic_write(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".agent-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def save_answer(task, answer):
    """Save only a single valid answer, beside task.json (not in workspace)."""
    if not isinstance(answer, str):
        return False
    text = answer.strip()
    if not text or len(text) > 2048 or "\n" in text or "\r" in text:
        return False
    if task["check"] == "exact_int" and re.fullmatch(r"[+-]?\d+", text) is None:
        return False
    if task["check"] == "unittest" and text != "DONE":
        return False
    atomic_write(task["directory"] / "answer.txt", text + "\n")
    return True


def make_prompt(task, max_bytes=100_000):
    """One bounded, visible workspace snapshot. Never include metadata or answers."""
    prompt = task["instruction"] + "\n\n"
    if task["task_type"] == "coding":
        prompt += ('Return ONLY a JSON object {"files": {"relative.py": "complete corrected source"}}. '
                   'Include changed source files only. Do not change tests. Do not run commands.\n')
    else:
        prompt += "Return only the final answer, on one line. Do not write files or explain.\n"
    root = task["workspace"]
    size = len(prompt.encode())
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in ("task.json", "answer.txt"):
            continue
        if path.suffix not in (".py", ".md", ".csv", ".txt", ".json") or not path.resolve().is_relative_to(root):
            continue
        block = "\nFILE " + path.relative_to(root).as_posix() + "\n" + path.read_text(encoding="utf-8") + "\nEND FILE\n"
        size += len(block.encode())
        if size > max_bytes:
            return None  # Never silently truncate the evidence.
        prompt += block
    return prompt if size <= max_bytes else None


def apply_code_answer(task, text):
    """Stage model files in scratch; copy back only after unchanged public tests pass."""
    import hashlib
    import shutil
    from submission.code_solver import run_tests, _passing
    try:
        files = json.loads(text)["files"]
        if not isinstance(files, dict) or not files or len(files) > 20:
            return False
        root = task["workspace"]
        if any(path.is_symlink() for path in root.rglob("*")):
            return False
        tests = {path.relative_to(root): hashlib.sha256(path.read_bytes()).digest()
                 for path in root.rglob("test*.py")}
        baseline = run_tests(root)
        if not tests or baseline is None:
            return False
        with tempfile.TemporaryDirectory(prefix="contest_model_patch_") as directory:
            scratch = Path(directory) / "workspace"
            shutil.copytree(root, scratch, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name, source in files.items():
                path = Path(name)
                target = (root / path).resolve()
                if (path.is_absolute() or ".." in path.parts or not target.is_relative_to(root)
                        or path.suffix != ".py" or path.name.startswith("test") or path.name == "conftest.py"
                        or not target.is_file() or not isinstance(source, str)):
                    return False
                compile(source, name, "exec")
                (scratch / path).write_text(source, encoding="utf-8")
            measured = run_tests(scratch)
            after = {path.relative_to(scratch): hashlib.sha256(path.read_bytes()).digest()
                     for path in scratch.rglob("test*.py")}
            if not _passing(measured) or measured["tests"] != baseline["tests"] or after != tests:
                return False
            for name in files:
                atomic_write(root / name, (scratch / name).read_text(encoding="utf-8"))
        return True
    except (KeyError, TypeError, ValueError, OSError, SyntaxError):
        return False
