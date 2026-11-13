"""Run with the installed SDK interpreter; no model endpoint is contacted."""
import asyncio
import contextlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

SDK = importlib.util.find_spec("openjiuwen") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(SDK, "Requires the existing openJiuwen SDK interpreter")
class LocalWorkflowChecks(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="contest-workflow-sdk-")
        cls.cwd = contextlib.chdir(cls.temporary.name)
        cls.cwd.__enter__()  # SDK import-time logs belong in this temporary directory.
        sys.path.insert(0, str(ROOT))
        from submission import workflow
        cls.module = workflow
        from openjiuwen.core.common.logging import LogManager
        from openjiuwen.core.common.logging.log_config import log_config
        cls.log_config, cls.log_snapshot = log_config, log_config.get_snapshot()
        log_config.load_from_dict({**cls.log_snapshot, "level": "CRITICAL", "loggers": {}})
        for logger in LogManager.get_all_loggers().values():
            logger.set_level("CRITICAL")

    @classmethod
    def tearDownClass(cls):
        cls.log_config.load_from_dict(cls.log_snapshot)
        sys.path.remove(str(ROOT))
        cls.cwd.__exit__(None, None, None)
        cls.temporary.cleanup()

    async def test_real_math_chain_through_native_graph(self):
        instruction = ("A warehouse starts with 81 boxes. It ships 6 boxes per day for 4 days. "
            "It receives 3 deliveries of 9 boxes each. Then the stock is split equally into 5 "
            "warehouses and it keeps one share (rounded down). How many boxes does it have at the end?")
        with tempfile.TemporaryDirectory() as directory:
            result = await self.module.run_local("math", instruction, Path(directory))
        self.assertEqual(result, {"status": "solved", "answer": "16",
                                  "reason": "integer_chain_fully_parsed"})

    async def test_real_search_and_unknown_keep_answer_field(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kb").mkdir()
            (workspace / "kb/record.md").write_text("# Ari Bell\nBorn: 1997\n")
            known = await self.module.run_local("search", "In which year was Ari Bell born?", workspace)
            missing = await self.module.run_local("search", "In which year was Missing Person born?", workspace)
        self.assertEqual((known["status"], known["answer"]), ("solved", "1997"))
        self.assertEqual(missing, {"status": "unknown", "answer": None, "reason": "missing_person"})

    async def test_real_csv_handler_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "sales.csv").write_text("product,region,units,price,month\n"
                "Lime,West,7,2,4\nLime,West,3,2,4\nPear,West,100,2,4\n")
            instruction = ("The file sales.csv has columns product, region, units, price, month "
                "(3 rows). How many units of product 'Lime' in month 4 were sold?")
            result = await self.module.run_local("data_analysis", instruction, workspace)
        self.assertEqual((result["status"], result["answer"]), ("solved", "10"))

    async def test_coding_dispatch_runs_sync_handler_off_event_loop(self):
        seen, event_loop_thread = [], threading.get_ident()

        def handler(instruction, workspace):
            seen.append((instruction, workspace, threading.get_ident()))
            return {"status": "unknown", "answer": None, "reason": "synthetic_handler"}

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("submission.code_solver.solve_code", side_effect=handler):
                result = await self.module.run_local("coding", "Run the public tests.", workspace)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][:2], ("Run the public tests.", workspace))
        self.assertNotEqual(seen[0][2], event_loop_thread)
        self.assertEqual(result, {"status": "unknown", "answer": None, "reason": "synthetic_handler"})

    async def test_independent_sessions_do_not_mix_parallel_results(self):
        with tempfile.TemporaryDirectory() as directory:
            results = await asyncio.gather(*(self.module.run_local("math",
                f"A warehouse starts with {number} boxes. How many boxes does it have at the end?",
                Path(directory)) for number in range(10, 15)))
        self.assertEqual([row["answer"] for row in results], [str(number) for number in range(10, 15)])

    async def test_unsupported_type_does_not_read_metadata_or_invoke_models(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "task.json").write_text('{"expected":"DO_NOT_READ"}')
            with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
                result = await self.module.run_local("unknown_type", "Anything", workspace)
        self.assertEqual(result, {"status": "unsupported", "answer": None,
                                  "reason": "unsupported_task_type"})


if __name__ == "__main__":
    unittest.main()
