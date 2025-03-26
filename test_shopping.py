import unittest
from shopping import *


class TestLoadData(unittest.TestCase):
    def setUp(self):
        evidence, labels = load_data('shopping.csv')
        self.test_data = [evidence[0]], [labels[0]]
        
    def test_load_data(self):
        expected_result = (
            [[0, 0.0, 0, 0.0, 1, 0.0,
              0.2, 0.2, 0.0, 0.0, 1, 
              1, 1, 1, 1, 1, 0]], [0]
        )
        self.assertEqual(self.test_data, expected_result)


class TestTrainModel(unittest.TestCase):
    def setUp(self):
        None

    def test_train_model(self):
        evidence, labels = load_data('shopping.csv')
        self.assertIsInstance(
            train_model(evidence, labels), KNeighborsClassifier
        )


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.evidence, self.labels = load_data('shopping.csv')

    def test_evaluate_100(self):
        expected_result = (1.0, 1.0)
        self.assertEqual(evaluate(self.labels, self.labels), expected_result)

    def test_evaluate_0(self):
        expected_result = (0.0, 0.0)
        not_labels = [1 if x == 0 else 0 for x in self.labels]
        self.assertEqual(evaluate(self.labels, not_labels), expected_result)


if __name__ == '__main__':
    unittest.main()