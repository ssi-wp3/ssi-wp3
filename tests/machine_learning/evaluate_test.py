from ssi.machine_learning.evaluate import evaluate
import unittest


class TestEvaluateFunctions(unittest.TestCase):
    def test_evaluate(self):
        self.assertEqual(evaluate(), "evaluate")
