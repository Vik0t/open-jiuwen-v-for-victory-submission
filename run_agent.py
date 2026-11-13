"""Contest entrypoint: exact handlers, then a compact openJiuwen agent loop."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from submission.budget import Budget
from submission.executor import protect_parent
from submission.task_io import load_tasks, save_answer, atomic_write


# Expected stopping cost from per-task means of old json_cues27 r1/r2 traces.
# Units: output tokens + input tokens / 10.
MEAN_COST = {
    "coding": 1205.8942307692307,
    "data_analysis": 497.12,
    "math": 392.06458333333336,
    "search": 448.506
}
SUCCESS_PROBABILITY = {
    "coding": 0.7798387096774194,
    "data_analysis": 0.6725,
    "math": 0.7991379310344828,
    "search": 0.6891666666666667
}
FALLBACK_COST = 425.1
FALLBACK_SUCCESS = 0.735

async def solve(tasks, budget, no_llm=False, local_address=None, execute_python=None):
    from submission.workflow import run_local
    rows, pending = [], []

    def record(task, result):
        row = {"task_id": task["task_id"], "reward": task["reward"], **result}
        if "answer" in row:
            answer = row.pop("answer")
            row["saved"] = answer is not None and save_answer(task, answer)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "steps"}, ensure_ascii=False), flush=True)
        atomic_write(task["directory"].parent / "agent-report.json", json.dumps({
            "spent_units": budget.spent_units, "reserved_units": budget.reserved_units,
            "budget_units": budget.limit_units, "budget_frozen": budget.frozen, "tasks": rows},
            ensure_ascii=False, indent=2) + "\n")

    for task in tasks:
        if task["reward"] == 0:
            record(task, {"status": "zero_reward_skipped", "saved": False, "method": "skipped"})
            continue
        # Blind AST mutations can overfit one public test. The model must also
        # consider intended behavior and regressions before submitting a repair.
        if task["task_type"] == "coding":
            pending.append(task)
            continue
        try:
            result = await run_local(task["task_type"], task["instruction"], task["workspace"])
        except Exception as error:
            result = {"status": "unsupported", "answer": None, "reason": type(error).__name__}
        if result["status"] == "unsupported":
            pending.append(task)
        else:
            record(task, dict(result, method="local"))

    if no_llm or not pending or not os.environ.get("LLM_BASE_URL") or not os.environ.get("LLM_API_KEY"):
        return rows
    from submission.compact import run_task
    from submission.executor import execute_python as default_executor
    from submission.gateway import Gateway
    execute_python = execute_python or default_executor
    # Queue estimate only: the Gateway still reserves every actual request.
    estimated_total_cost = sum(MEAN_COST.get(task["task_type"], FALLBACK_COST) for task in pending)
    scarce = budget.remaining_units / 10 < estimated_total_cost
    def priority(task):
        if not scarce:
            size = sum(path.stat().st_size for path in task["workspace"].rglob("*") if path.is_file())
            return (task["task_type"] == "coding", (size + 4000) / max(task["reward"], 1))
        kind = task["task_type"]
        value = task["reward"] * SUCCESS_PROBABILITY.get(kind, FALLBACK_SUCCESS)
        return (-value / MEAN_COST.get(kind, FALLBACK_COST), task["task_id"])

    async with Gateway(budget, os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], local_address=local_address) as gateway:
        for task in sorted(pending, key=priority):
            if budget.remaining_units == 0:
                break
            started = time.monotonic()
            try:
                result = await run_task(task, gateway, execute_python)
            except Exception as error:
                result = {"status": type(error).__name__, "saved": False}
            record(task, dict(result, method="compact_openjiuwen", seconds=round(time.monotonic()-started, 3)))
    return rows


def main():
    protect_parent()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=os.environ.get("TASKS"))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--local-address", help="Optional DeepCode route, unused by the judge")
    args = parser.parse_args()
    if args.tasks_dir is None:
        parser.error("Provide --tasks-dir or TASKS")
    tasks = load_tasks(args.tasks_dir.resolve())
    if not tasks:
        parser.error("No */task.json found")
    budget = Budget(os.environ.get("BUDGET", "0"))
    os.chdir("/tmp")
    try:
        asyncio.run(asyncio.wait_for(solve(tasks, budget, args.no_llm, args.local_address), 3500))
    except TimeoutError:
        print('{"status":"time_limit","answers_already_checkpointed":true}', flush=True)


if __name__ == "__main__":
    main()
