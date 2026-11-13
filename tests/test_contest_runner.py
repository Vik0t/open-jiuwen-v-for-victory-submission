import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from run_agent import solve
from submission.budget import Budget


class RunnerChecks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tasks = []
        for name in ("a", "b"):
            folder = Path(self.directory.name) / name
            folder.mkdir()
            self.tasks.append(dict(task_id=name, task_type="math", instruction="Compute a value",
                                   check="exact_int", reward=3, directory=folder, workspace=folder / "workspace"))

    async def test_pending_tasks_use_injected_executor_and_checkpoint_results(self):
        calls, executor = [], object()
        class Gateway:
            def __init__(self, *args, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        async def run_task(task, gateway, execute_python):
            calls.append((task["task_id"], execute_python))
            return {"status":"abstained", "saved":False}
        for task in self.tasks:
            task["workspace"].mkdir()
        modules = {"submission.workflow": SimpleNamespace(run_local=AsyncMock(return_value={"status":"unsupported", "answer":None})),
                   "submission.gateway": SimpleNamespace(Gateway=Gateway),
                   "submission.compact": SimpleNamespace(run_task=run_task)}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"LLM_BASE_URL":"https://unit.invalid", "LLM_API_KEY":"test"}), contextlib.redirect_stdout(io.StringIO()):
            rows = await solve(self.tasks, Budget(2000), execute_python=executor)
        self.assertEqual([item[0] for item in calls], ["a", "b"])
        self.assertTrue(all(item[1] is executor for item in calls))
        self.assertEqual(len(rows), 2)
        self.assertTrue((self.tasks[0]["directory"].parent / "agent-report.json").exists())

    async def test_unknown_local_does_not_request_model_and_solved_is_saved(self):
        local = AsyncMock(side_effect=[{"status":"unknown", "answer":None}, {"status":"solved", "answer":"21"}])
        modules = {"submission.workflow": SimpleNamespace(run_local=local),
                   "submission.gateway": None}
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            await solve(self.tasks, Budget(0))
        self.assertFalse((self.tasks[0]["directory"] / "answer.txt").exists())
        self.assertEqual((self.tasks[1]["directory"] / "answer.txt").read_text(), "21\n")


if __name__ == "__main__":
    unittest.main()
