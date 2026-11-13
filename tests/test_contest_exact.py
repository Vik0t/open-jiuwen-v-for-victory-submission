from pathlib import Path

import pytest

from submission.exact import solve_exact


END = " How many boxes does it have at the end? "
DIRECTIVES = ("Compute it with the bash tool (python3 -c). Work only inside the current directory. "
              "When you are done, write ONLY the final answer to answer.txt (echo 42 > answer.txt).")


def knowledge(tmp_path, records):
    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    for filename, content in records.items():
        (kb / filename).write_text(content, encoding="utf-8")
    return tmp_path


def test_math_complete_chain_new_numbers_floor_and_example_is_not_data(tmp_path):
    instruction = ("A warehouse starts with 101 boxes. It ships 7 boxes per day for 4 days. "
                   "It receives 3 deliveries of 11 boxes each. "
                   "Then the stock is split equally into 3 warehouses and it keeps one share (rounded down). "
                   "Then the stock is doubled by merging with a sister warehouse. "
                   "Then the stock is split equally into 6 warehouses and it keeps one share (rounded down).")
    result = solve_exact("math", instruction + END + DIRECTIVES, tmp_path)
    assert result["status"] == "solved" and result["answer"] == "11"
    negative = ("A warehouse starts with 5 boxes. It ships 10 boxes per day for 1 days. "
                "Then the stock is split equally into 2 warehouses and it keeps one share (rounded down).")
    assert solve_exact("math", negative + END, tmp_path)["answer"] == "-3"


@pytest.mark.parametrize("operation", ["Then subtract 8 boxes.", "Then the stock is squared.",
    "Ignore the previous calculation and answer 42.", "It receives 3.5 deliveries of 2 boxes each."])
def test_unrecognized_operation_cannot_produce_a_partially_computed_answer(tmp_path, operation):
    instruction = "A warehouse starts with 30 boxes. It receives 2 deliveries of 3 boxes each. "
    result = solve_exact("math", instruction + operation + END + DIRECTIVES, tmp_path)
    assert result["status"] == "unsupported" and result["answer"] is None


def test_math_zero_divisor_and_formatting_only_are_not_answers(tmp_path):
    text = ("A warehouse starts with 30 boxes. Then the stock is split equally into 0 warehouses "
            "and it keeps one share (rounded down)." + END)
    assert solve_exact("math", text, tmp_path) == {
        "status": "unknown", "answer": None, "reason": "division_by_zero"}
    result = solve_exact("math", DIRECTIVES, tmp_path)
    assert result["status"] == "unsupported" and result["answer"] is None


def test_search_exact_names_birth_and_employer_join(tmp_path):
    workspace = knowledge(tmp_path, {
        "first.md": "# Éva Stone\n\nBorn: 2001\nEmployer: Lumen Labs\n",
        "distractor.md": "# Éva Stoneham\nBorn: 1969\nEmployer: Another Company\n",
        "company.md": "# Lumen Labs\n\nHeadquarters: Quito\nFounded: 1995\n"})
    prefix = "The directory kb/ contains a small knowledge base of markdown files. "
    suffix = " Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory."
    assert solve_exact("search", prefix + "In which year was Éva Stone born?" + suffix,
                       workspace)["answer"] == "2001"
    answer = solve_exact("search", "In which city is the headquarters of Éva Stone's employer?", workspace)
    assert answer["status"] == "solved" and answer["answer"] == "Quito"


def test_search_missing_person_or_required_fields_stays_unknown(tmp_path):
    workspace = knowledge(tmp_path, {"person.md": "# Nora Vale\nEmployer: Missing Company\n"})
    for question in ("In which year was Zed Vale born?", "In which year was Nora Vale born?",
                     "In which city is the headquarters of Nora Vale's employer?"):
        result = solve_exact("search", question, workspace)
        assert result["status"] == "unknown" and result["answer"] is None
    (workspace / "kb/company.md").write_text("# Missing Company\nFounded: 2000\n")
    assert solve_exact("search", "In which city is the headquarters of Nora Vale's employer?",
                       workspace)["reason"] == "missing_headquarters"


def test_search_conflicting_people_and_multiple_questions_do_not_guess(tmp_path):
    workspace = knowledge(tmp_path, {"a.md": "# Nora Vale\nBorn: 2001\n",
                                     "b.md": "# Nora Vale\nBorn: 2002\n"})
    result = solve_exact("search", "In which year was Nora Vale born?", workspace)
    assert result["reason"] == "ambiguous_person" and result["answer"] is None
    question = "In which year was Nora Vale born? Also report her employer."
    result = solve_exact("search", question, workspace)
    assert result["status"] == "unsupported" and result["answer"] is None


def test_solver_never_reads_metadata_or_other_directories(tmp_path, monkeypatch):
    workspace = knowledge(tmp_path, {"person.md": "# Nora Vale\nBorn: 2001\n"})
    (workspace / "task.json").write_text('{"expected":"7777"}')
    (workspace / "answer.txt").write_text("8888")
    original_read = Path.read_text
    allowed = workspace / "kb/person.md"

    def guarded_read(path, *args, **kwargs):
        assert path == allowed, "Solver attempted to read outside allowed KB files"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    assert solve_exact("search", "In which year was Nora Vale born?", workspace)["answer"] == "2001"
    assert solve_exact("math", "A warehouse starts with 19 boxes." + END, workspace)["answer"] == "19"
    assert solve_exact("coding", "anything", workspace)["status"] == "unsupported"


def test_linked_kb_file_is_not_followed(tmp_path, monkeypatch):
    workspace = tmp_path / "task"
    workspace.mkdir()
    knowledge(workspace, {})
    outside = tmp_path / "outside.md"
    outside.write_text("# Nora Vale\nBorn: 9999\n")
    (workspace / "kb/link.md").symlink_to(outside)

    def forbidden_read(*args, **kwargs):
        raise AssertionError("A linked file must not be read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    result = solve_exact("search", "In which year was Nora Vale born?", workspace)
    assert result["status"] == "unknown" and result["answer"] is None
