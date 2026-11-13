"""Native openJiuwen workflow for local deterministic task handlers.

The runner changes cwd to its temporary directory before importing the SDK.
No task metadata, expected answers, model clients or network calls are used here.
"""
import asyncio
import logging
from pathlib import Path

from openjiuwen.core.common.logging import LogManager
from openjiuwen.core.common.logging.log_config import log_config
from openjiuwen.core.workflow import End, Start, Workflow, WorkflowComponent, create_workflow_session

from submission.exact import solve_exact


def _dispatch(task_type, instruction, workspace):
    workspace = Path(workspace)
    if task_type == "data_analysis":
        from submission.data_solver import solve_data
        return solve_data(instruction, workspace)
    if task_type == "coding":
        from submission.code_solver import solve_code
        return solve_code(instruction, workspace)
    return solve_exact(task_type, instruction, workspace)


class LocalSolver(WorkflowComponent):
    async def invoke(self, inputs, session, context):
        result = await asyncio.to_thread(_dispatch, inputs["task_type"],
                                         inputs["instruction"], inputs["workspace"])
        return {"result": result}


async def run_local(task_type, instruction, workspace: Path):
    """Execute Start -> local handler -> End and return the handler's dictionary.

    A nested result preserves answer=None through the SDK End component.
    Callers control concurrency and handler deadlines; to_thread is not a kill.
    """
    config = log_config.get_snapshot()
    config.update(level="CRITICAL", loggers={})
    log_config.load_from_dict(config)
    for logger in LogManager.get_all_loggers().values():
        logger.set_level(logging.CRITICAL)
    workflow = Workflow()
    workflow.set_start_comp("start", Start(), inputs_schema={
        "task_type": "${task_type}", "instruction": "${instruction}",
        "workspace": "${workspace}"})
    workflow.add_workflow_comp("solve", LocalSolver(), inputs_schema={
        "task_type": "${start.task_type}", "instruction": "${start.instruction}",
        "workspace": "${start.workspace}"})
    workflow.set_end_comp("end", End(), inputs_schema={"result": "${solve.result}"})
    workflow.add_connection("start", "solve")
    workflow.add_connection("solve", "end")
    output = await workflow.invoke({"task_type": task_type, "instruction": instruction,
                                    "workspace": str(workspace)}, create_workflow_session())
    return output.result["output"]["result"]
