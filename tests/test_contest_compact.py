import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from submission.budget import Budget
from submission.compact import run_task


class ScriptedGateway:
    def __init__(self, actions, status="ok"):
        self.budget = Budget(12000)
        self.actions, self.status, self.prompts = iter(actions), status, []

    async def answer(self, task_id, prompt, max_output, **kwargs):
        self.prompts.append(prompt)
        return {"status": self.status, "text": json.dumps(next(self.actions)), "task_id": task_id}


def execute_fixture(workspace, code, timeout=10):
    # Only fixed test programs, never model-generated code on the host.
    return subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=workspace,
                          capture_output=True, text=True, timeout=timeout)


class CompactTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        workspace = folder / "workspace"
        workspace.mkdir()
        self.task = dict(task_id="unit", task_type="math", instruction="Compute the sum",
                         check="exact_int", reward=5, directory=folder, workspace=workspace)

    async def test_python_saves_answer_without_another_model_call(self):
        gateway = ScriptedGateway([{"action": "python", "code": "submit_answer(sum(range(10)))"}])
        result = await run_task(self.task, gateway, execute_fixture)
        self.assertTrue(result["saved"])
        self.assertEqual(len(gateway.prompts), 1)
        self.assertEqual((self.task["directory"] / "answer.txt").read_text(), "45\n")

    async def test_completed_answer_survives_unknown_usage(self):
        gateway = ScriptedGateway([{"action": "answer", "answer": "21"}], status="unknown_usage")
        result = await run_task(self.task, gateway, execute_fixture)
        self.assertTrue(result["saved"])

    async def test_abstention_does_not_create_answer_file(self):
        gateway = ScriptedGateway([{"action": "abstain", "reason": "Missing denominator"}])
        result = await run_task(self.task, gateway, execute_fixture)
        self.assertEqual(result["status"], "abstained")
        self.assertFalse((self.task["directory"] / "answer.txt").exists())

    async def test_public_test_failure_cannot_be_submitted(self):
        self.task.update(task_type="coding", check="unittest")
        (self.task["workspace"] / "test_case.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n def test_bad(self): self.fail('regression')\n")
        gateway = ScriptedGateway([{"action": "answer", "answer": "DONE"}])
        result = await run_task(self.task, gateway, execute_fixture, max_steps=1)
        self.assertFalse(result["saved"])
        self.assertFalse((self.task["directory"] / "answer.txt").exists())

    async def test_changed_input_is_not_accepted(self):
        (self.task["workspace"] / "data.csv").write_text("value\n4\n")
        gateway = ScriptedGateway([{"action": "python", "code":
            "open('data.csv','w').write('value\\n100\\n'); submit_answer(100)"}])
        result = await run_task(self.task, gateway, execute_fixture, max_steps=1)
        self.assertFalse(result["saved"])
        self.assertIn("changed", result["steps"][0]["observation"]["error"])

    async def test_passing_edit_is_saved_on_last_step(self):
        self.task.update(task_type="coding", check="unittest")
        (self.task["workspace"] / "subject.py").write_text("value = 1\n")
        (self.task["workspace"] / "test_case.py").write_text(
            "import unittest\nfrom subject import value\nclass T(unittest.TestCase):\n def test_value(self): self.assertEqual(value, 2)\n")
        gateway = ScriptedGateway([{"action":"edit", "path":"subject.py", "before":"value = 1", "after":"value = 2"}])
        result = await run_task(self.task, gateway, execute_fixture, max_steps=1)
        self.assertTrue(result["saved"])
        self.assertEqual((self.task["directory"] / "answer.txt").read_text(), "DONE\n")

    async def test_new_analysis_output_does_not_change_provided_input(self):
        gateway = ScriptedGateway([{"action":"python", "code":"open('calculation.txt','w').write('21'); submit_answer(21)"}])
        self.assertTrue((await run_task(self.task, gateway, execute_fixture))["saved"])


if __name__ == "__main__":
    unittest.main()
