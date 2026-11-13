import tempfile
from pathlib import Path
import unittest

from submission.code_solver import solve_code


class CodeSolverTests(unittest.TestCase):
    def fixture(self, module, tests):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "mylib.py").write_text(module)
        (root / "test_mylib.py").write_text("import unittest\nfrom mylib import *\n" + tests)
        return root

    def test_two_generic_repairs_preserve_tests(self):
        root = self.fixture("def rearrange(s):\n return ' '.join(sorted(s.split()))\n"
                            "def accumulate(xs):\n s=0\n for x in xs: s-=x\n return s\n",
                            "class T(unittest.TestCase):\n"
                            " def test_order(self): self.assertEqual(rearrange('c a z'), 'z a c')\n"
                            " def test_sum(self): self.assertEqual(accumulate([8,-2,4]), 10)\n")
        test_bytes = (root / "test_mylib.py").read_bytes()
        result = solve_code("Fix the module until tests pass", root)
        self.assertEqual(result["answer"], "DONE")
        self.assertEqual(test_bytes, (root / "test_mylib.py").read_bytes())
        self.assertGreater(result["attempts"], 0)

    def test_compare_and_parity(self):
        root = self.fixture("def biggest(xs):\n best=xs[0]\n for x in xs:\n  if x<best: best=x\n return best\n"
                            "def evens(xs):\n return sum(1 for x in xs if x%2==1)\n",
                            "class T(unittest.TestCase):\n"
                            " def test_big(self): self.assertEqual(biggest([4,12,3]),12)\n"
                            " def test_even(self): self.assertEqual(evens([2,4,8,11]),3)\n")
        self.assertEqual(solve_code("Run tests", root)["status"], "solved")

    def test_failed_candidate_never_committed(self):
        root = self.fixture("def f(): return 11\n", "class T(unittest.TestCase):\n def test_x(self): self.assertEqual(f(),37)\n")
        before = (root / "mylib.py").read_bytes()
        self.assertEqual(solve_code("Run tests", root)["status"], "unsupported")
        self.assertEqual(before, (root / "mylib.py").read_bytes())

    def test_product_and_inclusive_boundary(self):
        root = self.fixture("def product(a,b): return a+b\n"
                            "def keep(xs): return [x for x in xs if x>0]\n",
                            "class T(unittest.TestCase):\n"
                            " def test_mul(self):\n  self.assertEqual(product(3,8),24)\n  self.assertEqual(product(-2,0),0)\n"
                            " def test_keep(self): self.assertEqual(keep([2,0,-7]),[2,0])\n")
        self.assertEqual(solve_code("Run tests", root)["status"], "solved")

    def test_zero_tests_and_skips_not_solved(self):
        for tests in ("", "@unittest.skip('skip')\nclass T(unittest.TestCase):\n def test_x(self): pass\n",
                      "class T(unittest.TestCase):\n @unittest.expectedFailure\n def test_x(self): self.fail()\n"):
            with self.subTest(tests=tests):
                root = self.fixture("def f(): return 1\n", tests)
                self.assertNotEqual(solve_code("Run tests", root)["status"], "solved")

    def test_attempt_limit_and_already_correct(self):
        root = self.fixture("def f(): return min([2,9])\n", "class T(unittest.TestCase):\n def test_x(self): self.assertEqual(f(),9)\n")
        result = solve_code("Run tests", root, max_attempts=0)
        self.assertEqual(result["attempts"], 0)
        self.assertNotEqual(result["status"], "solved")
        (root / "mylib.py").write_text("def f(): return 9\n")
        self.assertEqual(solve_code("Run tests", root)["status"], "solved")


if __name__ == "__main__":
    unittest.main()
