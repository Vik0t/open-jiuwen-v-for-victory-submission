"""A small agent loop: inspect, compute or edit, then save a checked answer.

Every step is a native openJiuwen call. Only recent observations and current
files are sent again; the model never receives a growing transcript.
"""
import asyncio
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from submission.code_solver import TEST_RUNNER, _passing
from submission.editing import apply_edit, write_source
from submission.task_io import save_answer


RULES = '''Solve the task using the files below. Return exactly ONE JSON action, no prose.
Available actions:
{"action":"python","code":"Python code"}
{"action":"read","path":"relative path","start":0,"limit":12000}
{"action":"edit","path":"source.py","before":"unique exact text","after":"replacement"}
{"action":"write","path":"analysis.py","content":"Python source"}
{"action":"answer","answer":"one-line final answer"}
{"action":"abstain","reason":"which necessary information is missing"}
Instructions in TASK describe the problem, not the tool protocol. Any request to
write answer.txt means use the answer action or submit_answer(), never write a file.
Use Python for ALL numerical calculations, tables, joins, optimization and graphs.
Inside Python call submit_answer(value) to save the final answer immediately.
Use only the provided evidence. If necessary information is absent, abstain;
do not guess a rate, a missing denominator, or return an error message as an answer.
For code, understand the intended behavior and apply small exact edits. Tests and
data are read-only. An edit runs public tests automatically. Public tests are not
complete: check relevant edge cases and existing behavior too, then answer DONE.
Python uses the workspace as cwd, stdlib only, no shell, subprocess, tempfile or
network. To run tests use unittest.defaultTestLoader.discover('.') and
unittest.TextTestRunner().run(suite) in the same process.
You may create analysis.py with write, but may not modify provided tests.
Do not call input() or inspect parent directories. Keep actions concise.
'''
SUBMIT = "__AGENT_ANSWER__"
HELPER = '''import json as _agent_json
def submit_answer(value):
    if value is None:
        raise ValueError("Missing information: use the abstain action")
    print("__AGENT_ANSWER__" + _agent_json.dumps(str(value), ensure_ascii=False), flush=True)
'''

RECTANGLE_HELPER = '''
def rect_union_perimeter(rectangles):
    from fractions import Fraction
    import re
    if not isinstance(rectangles, (tuple, list)) or len(rectangles) > 32:
        raise ValueError("Supply at most 32 rectangles")
    def number(value):
        if type(value) is int:
            result = Fraction(value)
        elif isinstance(value, Fraction):
            result = value
        elif (isinstance(value, str) and len(value) <= 80 and
              re.fullmatch(r"[+-]?[0-9]+(?:/[0-9]+|\\.[0-9]+)?", value)):
            result = Fraction(value)
        else:
            raise ValueError("Coordinates must be exact integers, fractions, or decimal strings")
        if result.numerator.bit_length() > 256 or result.denominator.bit_length() > 256:
            raise ValueError("Coordinate magnitude is too large")
        return result
    boxes = []
    for rectangle in rectangles:
        if not isinstance(rectangle, (tuple, list)) or len(rectangle) != 4:
            raise ValueError("Each rectangle must contain four coordinates")
        a, b, c, d = map(number, rectangle)
        x1, x2 = sorted((a, c))
        y1, y2 = sorted((b, d))
        if x1 == x2 or y1 == y2:
            raise ValueError("Rectangles must have positive area")
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        return Fraction(0)
    xs = sorted({x for x1, y1, x2, y2 in boxes for x in (x1, x2)})
    ys = sorted({y for x1, y1, x2, y2 in boxes for y in (y1, y2)})
    width, height = len(xs) - 1, len(ys) - 1
    if width * height > 4096:
        raise ValueError("Compressed grid is too large")
    covered = [[any(x1 <= xs[i] and xs[i+1] <= x2 and
                    y1 <= ys[j] and ys[j+1] <= y2 for x1, y1, x2, y2 in boxes)
                for j in range(height)] for i in range(width)]
    perimeter = Fraction(0)
    for i in range(width):
        for j in range(height):
            if covered[i][j]:
                dx, dy = xs[i+1] - xs[i], ys[j+1] - ys[j]
                if i == 0 or not covered[i-1][j]: perimeter += dy
                if i == width-1 or not covered[i+1][j]: perimeter += dy
                if j == 0 or not covered[i][j-1]: perimeter += dx
                if j == height-1 or not covered[i][j+1]: perimeter += dx
    return perimeter
'''


def has_rectangle_union_perimeter(task_type, instruction):
    """Offer only for explicitly requested axis-aligned union-boundary perimeter."""
    if task_type != "math" or not isinstance(instruction, str):
        return False
    text = " ".join(instruction.casefold().split())
    if re.search(r"\b(?:outer|outermost|exterior|external)\b|"
                 r"\b(?:ignore|exclude|excluding|except|without)\s+(?:(?:the|any)\s+)?holes?\b|"
                 r"\bholes?\s+(?:are\s+)?(?:excluded|ignored)\b|"
                 r"\b(?:not|no|never|avoid|exclude|ignore)\b(?:\s+\w+){0,4}\s+perimeter\b", text):
        return False
    return bool(re.search(r"\baxis[ -]+aligned\b", text)
                and re.search(r"\brectangles\b", text)
                and re.search(r"\bperimeter\s+of\s+(?:their|the)\s+union\b", text))


def protected_files(workspace):
    """Supplied tests and data must survive all actions byte for byte."""
    return {path.relative_to(workspace).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in workspace.rglob("*") if path.is_file()
            and (path.suffix != ".py" or path.name.startswith("test") or "tests" in path.parts)}


def unchanged(workspace, original):
    return all((workspace / name).is_file()
               and hashlib.sha256((workspace / name).read_bytes()).hexdigest() == digest
               for name, digest in original.items())


def read_file(workspace, name, start=0, limit=16000):
    relative = Path(name)
    path = workspace / relative
    if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
            or not path.resolve().is_relative_to(workspace) or not path.is_file()
            or path.name in {"task.json", "answer.txt"}):
        raise ValueError("Read a regular workspace file by relative path")
    if type(start) is not int or start < 0 or type(limit) is not int or not 1 <= limit <= 20000:
        raise ValueError("Invalid character range")
    text = path.read_bytes().decode("utf-8")
    return {"path": relative.as_posix(), "start": start, "total_chars": len(text), "text": text[start:start+limit]}


def remember_file(files, item):
    """Keep a recent, consistent excerpt per file within 24,000 characters."""
    old = files.pop(item["path"], None)
    if old and old["total_chars"] == item["total_chars"]:
        left, right = sorted((old, item), key=lambda part: part["start"])
        overlap = left["start"] + len(left["text"]) - right["start"]
        if overlap >= 0:
            shared = min(overlap, len(right["text"]))
            offset = right["start"] - left["start"]
            if left["text"][offset:offset+shared] == right["text"][:shared]:
                merged = left["text"] + right["text"][overlap:]
                if len(merged) <= 20000:
                    item = {**item, "start": left["start"], "text": merged}
    files[item["path"]] = item
    while sum(len(part["text"]) for part in files.values()) > 24000:
        del files[next(iter(files))]


def file_index(root):
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.name not in {"task.json", "answer.txt"}]


def deferred_licenses(root, index, instruction):
    """Only defer recognized boilerplate when all other public text is safe to inspect."""
    cue = re.compile(r"licen[cs]|copyright|compliance|\bcopying\b|\bnotice\b", re.I)
    names = [name for name in index if Path(name).name.casefold() in {"license", "license.txt"}]
    if not names or cue.search(instruction) or len(index) > 64:
        return set()
    try:
        paths = [(name, root / name) for name in index]
        if sum(path.stat().st_size for _, path in paths) > 512000:
            return set()
        texts = {name: path.read_bytes().decode("utf-8") for name, path in paths}
        readme = texts.get("README.md")
        if readme is not None and len(readme) > 20000:
            return set()
        for name, text in texts.items():
            if name not in names and (cue.search(name) or cue.search(text)):
                return set()
        # Whole-file hashes reject a standard-looking heading with extra terms.
        # Unrecognized variants stay visible; no generic license is assumed safe.
        known = {
            "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf",  # PSF
            "554ba13996850c66db2754c31fe2ed0569cce43515966de10a1a6ef5d1cf4ed0",  # MIT
            "b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1",  # Apache 2.0
        }
        return set(names) if all(hashlib.sha256(texts[name].encode("utf-8")).hexdigest() in known
                                 for name in names) else set()
    except (OSError, UnicodeError, ValueError):
        return set()


def parse_action(text, rectangle_enabled=False):
    text = (text or "").strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    result = json.loads(text)
    allowed = {"python", "read", "edit", "write", "answer", "abstain"}
    if rectangle_enabled:
        allowed.add("rect_union_answer")
    if not isinstance(result, dict) or result.get("action") not in allowed:
        raise ValueError("Return one valid JSON action from the available list")
    if result["action"] == "rect_union_answer":
        def unique_object(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("Duplicate JSON keys are not allowed")
                obj[key] = value
            return obj
        def no_constant(value):
            raise ValueError("Non-finite JSON numbers are not allowed")
        result = json.loads(text, object_pairs_hook=unique_object, parse_constant=no_constant)
    return result


def _typed_rectangles(action):
    """Validate JSON-only data before it can become a Python literal."""
    if set(action) != {"action", "rectangles"}:
        raise ValueError("Rectangle action must contain only action and rectangles")
    rectangles = action["rectangles"]
    if not isinstance(rectangles, list) or not 1 <= len(rectangles) <= 32:
        raise ValueError("Provide 1-32 rectangles as arrays")
    for rectangle in rectangles:
        if not isinstance(rectangle, list) or len(rectangle) != 4:
            raise ValueError("Each rectangle must be an array of four coordinates")
        exact = []
        for value in rectangle:
            if type(value) is int:
                number = Fraction(value)
            elif (type(value) is str and len(value) <= 80
                  and re.fullmatch(r"[+-]?[0-9]+(?:/[0-9]+|\.[0-9]+)?", value)):
                try:
                    number = Fraction(value)
                except (ValueError, ZeroDivisionError):
                    raise ValueError("Invalid exact coordinate") from None
            else:
                raise ValueError("Coordinates must be JSON integers or exact numeric strings")
            if number.numerator.bit_length() > 256 or number.denominator.bit_length() > 256:
                raise ValueError("Coordinate magnitude is too large")
            exact.append(number)
        if exact[0] == exact[2] or exact[1] == exact[3]:
            raise ValueError("Rectangles must have positive area")
    return rectangles


async def run_task(task, gateway, execute_python, max_steps=8):
    root, history, events = task["workspace"].resolve(), [], []
    rules = RULES if task["task_type"] != "math" else RULES.replace("stdlib only", "stdlib plus installed SymPy") + (
        "\nFor math prefer exact SymPy Rational, equations, polynomials and number theory. For rational LP use sympy.solvers.simplex.lpmax with explicit bounds; its variables are free. Avoid linprog for free/negative variables in this version. "
        "Derive equations from requirements and verify returned constraints; SymPy does not validate the model.\n")
    rectangle_helper = has_rectangle_union_perimeter(task["task_type"], task["instruction"])
    if rectangle_helper:
        rules += ("\nFor axis-aligned rectangle unions, Python has rect_union_perimeter(rectangles): "
                  "pass a list of (x1,y1,x2,y2) opposite-corner coordinates, as integers, Fraction, "
                  "or exact decimal/fraction strings. It returns a Fraction for all union boundaries "
                  "including holes. Verify the inputs and call submit_answer(str(result)) if appropriate.\n")
        rules += ('If coordinates are all known, you may instead return '
                  '{"action":"rect_union_answer","rectangles":[[x1,y1,x2,y2],...]}. '
                  'Coordinates must be JSON integers or exact decimal/fraction strings, not floats. '
                  'If required coordinates are missing, abstain.\n')
    if task["task_type"] == "coding":
        rules += ('\nFor several fixes in ONE file, use {"action":"edit","path":"source.py",'
                  '"changes":[{"before":"exact old fragment A","after":"replacement A"},'
                  '{"before":"exact old fragment B","after":"replacement B"}]}. '
                  'Give 1-8 replacements in order. They are validated in memory and written together. '
                  'Fix all explicitly requested behavior in that file before its public tests run.\n')
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        return {"status": "invalid_workspace", "saved": False, "steps": events}
    original = protected_files(root)
    readme_spec = ""
    if task["task_type"] == "coding":
        try:
            readme = read_file(root, "README.md", limit=20000)
            readme_spec = "\nRequired behavior from README.md:\n"
            if readme["total_chars"] > len(readme["text"]):
                readme_spec += (f"[PARTIAL: characters [0,{len(readme['text'])}) of {readme['total_chars']}. "
                                "Read the remaining README.md before treating it as the full contract.]\n")
            readme_spec += readme["text"]
        except (ValueError, UnicodeError, OSError):
            pass
    index = file_index(root)
    deferred = deferred_licenses(root, index, task["instruction"]) if task["task_type"] == "coding" else set()
    files, available = {}, 18000
    if task["task_type"] == "coding":
        from submission.named_initial import initial_range
        try:
            focused = initial_range(root, index, task["instruction"])
            if focused:
                item = read_file(root, *focused)
                remember_file(files, item)
                available -= len(item["text"])
        except (OSError, UnicodeError, ValueError, SyntaxError, RecursionError):
            pass
    for name in index:
        if available <= 0:
            break
        if name in deferred:
            continue
        if name in files:
            continue
        try:
            item = read_file(root, name, limit=min(available, 10000))
            remember_file(files, item)
            available -= len(item["text"])
        except (ValueError, UnicodeError, OSError):
            pass

    async def execute(code):
        result = await asyncio.to_thread(execute_python, root, code, 10)
        if not unchanged(root, original):
            raise ValueError("A provided test or data file was changed; submission rejected")
        return result

    async def tests():
        result = await execute(TEST_RUNNER)
        lines = [line[len("__TEST_RESULT__"):] for line in result.stdout.splitlines()
                 if line.startswith("__TEST_RESULT__")]
        summary = json.loads(lines[0]) if len(lines) == 1 else None
        passed = bool(summary and summary["tests"] > 0 and result.returncode == 0 and _passing(summary))
        return {"passed": passed, "summary": summary, "stderr": result.stderr[-3500:]}

    status = "step_limit"
    started_units = gateway.budget.spent_units
    task_allowance = max(15000, gateway.budget.limit_units // 4)
    for step in range(max_steps):
        if gateway.budget.spent_units - started_units >= task_allowance:
            status = "task_budget_limit"
            break
        prompt = "TASK:\n" + task["instruction"] + readme_spec
        prompt += f"\nActions remaining including this one: {max_steps-step}."
        if max_steps-step <= 2:
            prompt += " Save a verified result if ready; otherwise fix the demonstrated issue or abstain if evidence is missing. Do not invent an answer to meet the limit."
        prompt += "\nTask type: " + task["task_type"] + "\nFiles: " + json.dumps(index)
        prompt += "\nCurrent file excerpts:\n" + json.dumps(list(files.values()), ensure_ascii=False)
        prompt += "\nRecent actions and results:\n" + json.dumps(history[-2:], ensure_ascii=False)
        # Reservation uses request bytes as a conservative token bound. Leave
        # enough room for the envelope, instead of abandoning the budget tail.
        input_bound = len(json.dumps(prompt + rules, ensure_ascii=False).encode()) + 2500
        cap = min(1536, (gateway.budget.remaining_units - input_bound) // 10)
        if cap < 128:
            status = "budget_unavailable"
            break
        for attempt in range(2):
            response = await gateway.answer(task["task_id"], prompt, max_output=cap, system_prompt=rules,
                                            json_mode=task["task_type"] == "coding")
            event = {"step": step+1, "attempt": attempt+1, **response}
            events.append(event)  # Final content is retained; hidden reasoning is not.
            if response["status"] != "budget_unavailable" or not response.get("input_bound"):
                break
            smaller = (gateway.budget.remaining_units - response["input_bound"]) // 10 - 2
            if not 128 <= smaller < cap:
                break
            cap = smaller  # The denied request was not sent and cost no tokens.
        if response["status"] not in {"ok", "unknown_usage"} or not response.get("text"):
            status = "empty_action" if response["status"] == "ok" else response["status"]
            break
        try:
            action = parse_action(response["text"], rectangle_enabled=rectangle_helper)
            kind = action["action"]
            if kind == "abstain":
                status = "abstained"
                event["observation"] = action.get("reason", "")
                break
            if kind == "answer" or (kind == "write" and action.get("path") == "answer.txt"):
                answer = action.get("answer") if kind == "answer" else action.get("content")
                verified = task["task_type"] != "coding" or (await tests())["passed"]
                if verified and unchanged(root, original) and save_answer(task, answer):
                    return {"status": "completed", "saved": True, "steps": events}
                observation = {"error": "Answer format invalid or coding tests do not pass"}
            elif kind == "rect_union_answer":
                rectangles = _typed_rectangles(action)
                # Only validated integers and numeric strings enter the fixed code template.
                code = HELPER + RECTANGLE_HELPER + "\nsubmit_answer(str(rect_union_perimeter(" + repr(rectangles) + ")))"
                result = await execute(code)
                observation = {"returncode": result.returncode, "stdout": result.stdout[-8000:], "stderr": result.stderr[-3500:]}
                answers = [json.loads(line[len(SUBMIT):]) for line in result.stdout.splitlines() if line.startswith(SUBMIT)]
                if result.returncode == 0 and len(answers) == 1 and isinstance(answers[0], str) and save_answer(task, answers[0]):
                    event["observation"] = observation
                    return {"status": "completed", "saved": True, "steps": events}
                observation["error"] = "Rectangle calculation or answer validation failed"
            elif kind == "read":
                observation = read_file(root, action["path"], action.get("start", 0), action.get("limit", 12000))
                remember_file(files, observation)
            elif kind in {"edit", "write"}:
                if task["task_type"] != "coding" and kind == "edit":
                    raise ValueError("Input data are read-only")
                observation = (apply_edit(root, action["path"], action.get("before"), action.get("after"),
                                          changes=action.get("changes"))
                               if kind == "edit" else write_source(root, action["path"], action["content"]))
                if observation["success"] and not observation["changed"]:
                    observation["error"] = "No source bytes changed: before and after are identical. Choose a different edit or inspect the exact source text; repeating this action cannot fix the failing test."
                if observation["success"] and observation["changed"]:
                    index = file_index(root)
                    files.pop(Path(action["path"]).as_posix(), None)  # Discard old source before refreshing its prefix.
                    remember_file(files, read_file(root, action["path"], limit=10000))
                    if task["task_type"] == "coding":
                        observation["public_tests"] = await tests()
                        if observation["public_tests"]["passed"] and save_answer(task, "DONE"):
                            event["observation"] = observation
                            return {"status": "completed", "saved": True, "steps": events}
            else:
                code = action["code"]
                if not isinstance(code, str) or len(code) > 20000:
                    raise ValueError("Python code must be at most 20000 characters")
                before = {p: p.read_bytes() for p in root.rglob("*.py")}
                result = await execute(HELPER + (RECTANGLE_HELPER if rectangle_helper else "") + "\n" + code)
                observation = {"returncode": result.returncode, "stdout": result.stdout[-8000:], "stderr": result.stderr[-3500:]}
                changed = [p for p in root.rglob("*.py") if before.get(p) != p.read_bytes()]
                removed = [p for p in before if not p.exists()]
                if changed or removed:
                    index = file_index(root)
                for path in removed + changed:
                    files.pop(path.relative_to(root).as_posix(), None)
                for path in changed:
                    name = path.relative_to(root).as_posix()
                    remember_file(files, read_file(root, name, limit=10000))
                if changed or removed:
                    if task["task_type"] == "coding":
                        observation["public_tests"] = await tests()
                        if result.returncode == 0 and observation["public_tests"]["passed"] and save_answer(task, "DONE"):
                            event["observation"] = observation
                            return {"status": "completed", "saved": True, "steps": events}
                answers = [json.loads(line[len(SUBMIT):]) for line in result.stdout.splitlines() if line.startswith(SUBMIT)]
                if result.returncode == 0 and len(answers) == 1:
                    verified = task["task_type"] != "coding" or (await tests())["passed"]
                    if verified and save_answer(task, answers[0]):
                        event["observation"] = observation
                        return {"status": "completed", "saved": True, "steps": events}
            event["observation"] = observation
            if kind == "read":
                history.append({"action": action, "result": "Read completed"})
            else:
                history.append({"action": action, "result": observation})
        except (ValueError, KeyError, TypeError, OSError) as error:
            event["observation"] = {"error": str(error)[:1000]}
            history.append({"result": event["observation"]})
    return {"status": status, "saved": False, "steps": events}
