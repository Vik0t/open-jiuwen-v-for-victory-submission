"""Conservative exact solvers for observed math/search instruction grammars.

No task metadata, expected answers, shell commands or model calls are read/run.
An unrecognized operation/question never produces a partial guessed answer.
"""
from pathlib import Path
import re


def _result(status, answer=None, reason=""):
    return {"status": status, "answer": answer, "reason": reason}


def _problem(instruction):
    # Numbers in output examples (such as echo 42) are not problem inputs.
    return re.split(r"\b(?:Compute\b|Work only\b|When you are done\b)",
                    instruction, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _math(instruction):
    text = " ".join(_problem(instruction).split())
    match = re.fullmatch(
        r"A warehouse starts with (-?\d+) boxes\.\s*(.*?)\s*"
        r"How many boxes does it have at the end\?", text, re.IGNORECASE)
    if match is None:
        return _result("unsupported", reason="unrecognized_math_statement")
    value, remaining = int(match[1]), match[2].strip()
    operations = (
        (r"(?:Then )?It ships (\d+) boxes per day for (\d+) days\.",
         lambda number, boxes, days: number - boxes * days),
        (r"(?:Then )?It receives (\d+) deliveries of (\d+) boxes each\.",
         lambda number, deliveries, boxes: number + deliveries * boxes),
        (r"Then the stock is split equally into (\d+) warehouses and it keeps one share \(rounded down\)\.",
         lambda number, shares: number // shares),
        (r"Then the stock is doubled by merging with a sister warehouse\.",
         lambda number: number * 2),
    )
    while remaining:
        for pattern, operation in operations:
            step = re.match(pattern, remaining, re.IGNORECASE)
            if step is not None:
                value = operation(value, *(int(argument) for argument in step.groups()))
                remaining = remaining[step.end():].strip()
                break
        else:
            return _result("unsupported", reason="unrecognized_math_operation")
    return _result("solved", str(value), "integer_chain_fully_parsed")


def _name(text):
    return " ".join(text.split()).casefold()


def _search(instruction, workspace):
    text = " ".join(_problem(instruction).split())
    text = re.sub(r"^The directory kb/ contains a small knowledge base of markdown files\.\s*",
                  "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Use the file tools or bash \(ls, grep, cat\) to find it\.$",
                  "", text, flags=re.IGNORECASE)
    born = re.fullmatch(r"In which year was (.+?) born\?", text, re.IGNORECASE)
    headquarters = re.fullmatch(r"In which city is the headquarters of (.+?)[’']s employer\?",
                               text, re.IGNORECASE)
    query = born or headquarters
    if query is None:
        return _result("unsupported", reason="unrecognized_search_question")

    kb = Path(workspace) / "kb"
    if not kb.is_dir() or kb.is_symlink():
        return _result("unknown", reason="knowledge_base_missing_or_linked")
    records = {}
    for path in sorted(kb.glob("*.md")):
        if path.is_symlink():
            return _result("unknown", reason="linked_knowledge_file")
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        heading = re.match(r"\s*#\s+([^\n]+)", content)
        if heading is None:
            continue
        fields = {}
        for line in content.splitlines():
            field = re.fullmatch(r"(Born|Employer|Headquarters):\s*(.+?)\s*", line, re.IGNORECASE)
            if field is not None:
                key, value = field[1].casefold(), field[2]
                if key in fields and fields[key] != value:
                    return _result("unknown", reason="conflicting_knowledge_field")
                fields[key] = value
        records.setdefault(_name(heading[1]), []).append(fields)

    people = records.get(_name(query[1]), [])
    if len(people) != 1:
        return _result("unknown", reason="missing_person" if not people else "ambiguous_person")
    person = people[0]
    if born:
        year = person.get("born", "")
        if re.fullmatch(r"\d{4}", year) is None:
            return _result("unknown", reason="birth_year_missing_or_invalid")
        return _result("solved", year, "birth_year_from_knowledge_base")
    employer = person.get("employer")
    if employer is None:
        return _result("unknown", reason="missing_employer")
    companies = records.get(_name(employer), [])
    if len(companies) != 1:
        return _result("unknown", reason="missing_company" if not companies else "ambiguous_company")
    city = companies[0].get("headquarters")
    if city is None:
        return _result("unknown", reason="missing_headquarters")
    return _result("solved", city, "employer_headquarters_from_knowledge_base")


def solve_exact(task_type, instruction, workspace: Path):
    """Solve supported visible statements, otherwise return an explicit refusal.

    Math reads no files. Search reads only direct, non-symlink kb/*.md files.
    This function neither reads task.json nor writes answer.txt.
    """
    if not isinstance(instruction, str):
        return _result("unsupported", reason="instruction_must_be_text")
    try:
        if task_type == "math":
            return _math(instruction)
        if task_type == "search":
            return _search(instruction, workspace)
    except ZeroDivisionError:
        return _result("unknown", reason="division_by_zero")
    except (OSError, UnicodeError, ValueError):
        return _result("unknown", reason="invalid_or_unreadable_input")
    return _result("unsupported", reason="unsupported_task_type")
