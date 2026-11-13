import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from submission.executor import MAX_OUTPUT_BYTES, execute_python
from submission.code_solver import TEST_RUNNER


IN_CONTAINER = sys.platform == 'linux' and Path('/.dockerenv').is_file()


class ExecutorHostTests(unittest.TestCase):
    def test_refuses_host_execution_before_starting_subprocess(self):
        with patch('submission.executor.sys.platform', 'darwin'), \
                patch('submission.executor.subprocess.run') as run:
            result = execute_python('/tmp', 'raise RuntimeError("must not execute")')
        self.assertIsInstance(result, subprocess.CompletedProcess)
        self.assertEqual(result.returncode, 126)
        self.assertIn('container', result.stderr)
        run.assert_not_called()


@unittest.skipUnless(IN_CONTAINER, 'Real execution requires offline Docker; host refusal is tested above')
class ExecutorContainerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()

    def execute(self, code, timeout=3):
        return execute_python(self.workspace, code, timeout)

    def test_reads_workspace_imports_stdlib_and_module_and_writes_ordinary_file(self):
        (self.workspace / 'values.csv').write_text('value\n2\n5\n')
        (self.workspace / 'helper.py').write_text('factor = 3\n')
        code = ('import csv, json, math, helper\n'
                'with open("values.csv") as f: total = sum(int(r["value"]) for r in csv.DictReader(f))\n'
                'with open("calculation.txt", "w") as f: f.write(str(total * helper.factor))\n'
                'print(json.dumps({"answer": total * helper.factor, "root": math.isqrt(81)}))')
        result = self.execute(code)
        self.assertEqual(result.returncode, 0, result)
        self.assertEqual(json.loads(result.stdout), {'answer': 21, 'root': 9})
        self.assertEqual((self.workspace / 'calculation.txt').read_text(), '21')
        self.assertFalse((self.workspace / '__pycache__').exists())

    def test_environment_does_not_include_parent_secrets(self):
        with patch.dict(os.environ, {'EXECUTOR_TEST_SECRET': 'never forward this'}):
            result = self.execute('import os; print(os.environ.get("EXECUTOR_TEST_SECRET"))')
        self.assertEqual(result.returncode, 0, result)
        self.assertEqual(result.stdout.strip(), 'None')

    def test_unittest_mock_import_and_patch_work_in_provided_tests(self):
        (self.workspace / 'test_example.py').write_text(
            'import os, unittest\nfrom unittest.mock import patch\n'
            'class Example(unittest.TestCase):\n'
            ' def test_patch(self):\n'
            '  with patch("os.getcwd", return_value="patched"):\n'
            '   self.assertEqual(os.getcwd(), "patched")\n')
        result = self.execute(TEST_RUNNER)
        self.assertEqual(result.returncode, 0, result)
        summary = json.loads(next(line.removeprefix('__TEST_RESULT__')
                                  for line in result.stdout.splitlines()
                                  if line.startswith('__TEST_RESULT__')))
        self.assertEqual(summary['tests'], 1)
        self.assertEqual(summary['failures'], 0)

    def test_cannot_write_protected_paths_or_escape_through_symlink(self):
        outside = self.root / 'outside.txt'
        outside.write_text('keep')
        (self.workspace / 'alias.txt').symlink_to(outside)
        names = ('../outside.txt', 'alias.txt', 'test_solver.py', 'solver_test.py',
                 'conftest.py', 'task.json', 'answer.txt', 'tests/helper.py',
                 'metadata/info.txt', 'answers/result.txt', '.hidden')
        for name in names:
            with self.subTest(path=name):
                result = self.execute(f'from pathlib import Path; Path({name!r}).write_text("bad")')
                self.assertNotEqual(result.returncode, 0, result)
                self.assertIn('PermissionError', result.stderr)
        self.assertEqual(outside.read_text(), 'keep')

    def test_cannot_rename_delete_link_or_open_raw_descriptor_to_bypass_protection(self):
        (self.workspace / 'test_solver.py').write_text('keep')
        (self.workspace / 'ordinary.py').write_text('replacement')
        codes = ('import os; os.remove("test_solver.py")',
                 'import os; os.replace("ordinary.py", "test_solver.py")',
                 'import os; os.link("test_solver.py", "alias.py")',
                 'import os; os.open("tests", os.O_RDONLY)',
                 'import os; os.chmod("test_solver.py", 0o777)')
        for code in codes:
            with self.subTest(code=code):
                result = self.execute(code)
                self.assertNotEqual(result.returncode, 0, result)
                self.assertIn('PermissionError', result.stderr)
        self.assertEqual((self.workspace / 'test_solver.py').read_text(), 'keep')

    def test_network_subprocesses_native_access_and_signals_are_rejected(self):
        codes = ('import socket; socket.socket()',
                 'import os; os.system("true")',
                 'import subprocess; subprocess.run(["true"])',
                 'import os; os.fork()',
                 'import ctypes',
                 'import os; os.kill(os.getppid(), 0)')
        for code in codes:
            with self.subTest(code=code):
                result = self.execute(code)
                self.assertNotEqual(result.returncode, 0, result)
                self.assertIn('PermissionError', result.stderr)

    def test_private_parent_files_cannot_be_read(self):
        (self.root / 'task.json').write_text('{"expected_answer": 123}')
        result = self.execute('print(open("../task.json").read())')
        self.assertNotEqual(result.returncode, 0, result)
        self.assertNotIn('expected_answer', result.stdout)

    def test_main_module_cannot_replace_the_closed_over_path_policy(self):
        (self.root / 'task.json').write_text('{"private_sentinel": 123}')
        code = ('import __main__\nfrom pathlib import Path\n'
                'assert not hasattr(__main__, "root")\n'
                'assert not hasattr(__main__, "check_path")\n'
                '__main__.root = Path("/")\n'
                '__main__.check_path = lambda *args: None\n'
                'print(open("../task.json").read())')
        result = self.execute(code)
        self.assertNotEqual(result.returncode, 0, result)
        self.assertIn('PermissionError', result.stderr)
        self.assertNotIn('private_sentinel', result.stdout)

    def test_timeout_is_reported_and_captured_output_is_bounded(self):
        result = self.execute('import time; print("started", flush=True); time.sleep(10)', timeout=0.2)
        self.assertEqual(result.returncode, 124, result)
        self.assertEqual(result.stdout.strip(), 'started')
        self.assertIn('timeout', result.stderr)
        result = self.execute('print("x" * 20000)')
        self.assertEqual(result.returncode, 0, result)
        self.assertEqual(len(result.stdout.encode()), MAX_OUTPUT_BYTES)

    def test_memory_limit_prevents_large_allocation(self):
        result = self.execute('value = bytearray(700 * 1024**2)')
        self.assertNotEqual(result.returncode, 0, result)
        self.assertIn('MemoryError', result.stderr)

    def test_long_diagnostics_preserve_final_answer_and_error_tail(self):
        marker = '__AGENT_ANSWER__"21"'
        code = ('import sys\nprint("x" * 9000)\n'
                f'print({marker!r})\n'
                'print("y" * 9000 + "final error", file=sys.stderr)')
        result = self.execute(code)
        self.assertEqual(result.returncode, 0, result)
        self.assertEqual(len(result.stdout.encode()), MAX_OUTPUT_BYTES)
        self.assertEqual(len(result.stderr.encode()), MAX_OUTPUT_BYTES)
        self.assertTrue(result.stdout.endswith(marker + '\n'))
        self.assertTrue(result.stderr.endswith('final error\n'))

    def test_invalid_inputs_are_rejected_without_execution(self):
        for timeout in (0, -1, float('inf'), float('nan'), 61, '3'):
            self.assertEqual(execute_python(self.workspace, 'print(1)', timeout).returncode, 126)
        self.assertEqual(execute_python(self.workspace, None).returncode, 126)
        self.assertEqual(execute_python(self.root / 'missing', 'print(1)').returncode, 126)
        alias = self.root / 'alias'
        alias.symlink_to(self.workspace, target_is_directory=True)
        self.assertEqual(execute_python(alias, 'print(1)').returncode, 126)


if __name__ == '__main__':
    unittest.main()
