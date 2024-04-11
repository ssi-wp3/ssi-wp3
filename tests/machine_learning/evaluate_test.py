from ssi.machine_learning.evaluate import ConfusionMatrix
import unittest
import numpy as np


class ConfusionMatrixTest(unittest.TestCase):
    def test_confusion_matrix_label_mapping(self):
        count_matrix = np.array([[1, 2], [3, 4]])
        label_mapping = {"a": 0, "b": 1}
        confusion_matrix = ConfusionMatrix(count_matrix, label_mapping)
        self.assertEqual(label_mapping, confusion_matrix.label_mapping)
