import unittest
import numpy as np
from comparers import ModelComparer
from sklearn.dummy import DummyClassifier

from difference_models import MethodUndefinedException
from explainers import generate_shap_explanations


def make_binary_classification():
    class_names = ['False', 'True']
    feature_names = ['x1', 'x2', 'x3', 'x4']
    n_instances = 10
    X = np.repeat(np.arange(n_instances).reshape((n_instances, 1)), len(feature_names), axis=1).astype(float)
    y = X[:, 0] >= 5
    return X, y, class_names, feature_names


def make_multiclass_classification():
    class_names = ['c1', 'c2', 'c3']
    feature_names = ['x1', 'x2', 'x3', 'x4']
    n_instances = 10
    X = np.repeat(np.arange(n_instances).reshape((n_instances, 1)), len(feature_names), axis=1)
    y = np.mod(X[:, 0], len(class_names))
    X = X.astype(float)
    return X, y, class_names, feature_names


def make_simple_comparer(X, y, feature_names):
    c1 = DummyClassifier()
    c1.fit(X, y)
    c2 = DummyClassifier(strategy='uniform')
    c2.fit(X, y)
    comparer = ModelComparer(c1, c2, feature_names)
    comparer.fit(X, y)
    return comparer


class TestExplainersBinaryTask(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y, self.indiv_class_names, self.feature_names = make_binary_classification()
        self.comparer = make_simple_comparer(self.X, self.y, self.feature_names)

    def test_generate_indiv_shap_values_should_succeed(self):
        shap_values = generate_shap_explanations(self.comparer, self.X, explanation_type='indiv')
        self.assertEqual(shap_values.shape, (self.X.shape[0], len(self.feature_names), 2 * len(self.indiv_class_names)))
        self.assertListEqual(shap_values.feature_names, self.feature_names)
        self.assertListEqual(shap_values.output_names, ['A.False', 'A.True', 'B.False', 'B.True'])

    def test_generate_bin_diff_shap_values_should_succeed(self):
        shap_values = generate_shap_explanations(self.comparer, self.X, explanation_type='bin_diff')
        self.assertEqual(shap_values.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertListEqual(shap_values.feature_names, self.feature_names)
        self.assertEqual(shap_values.output_names, 'different')

    def test_generate_mclass_diff_shap_values_should_succeed(self):
        shap_values = generate_shap_explanations(self.comparer, self.X, explanation_type='mclass_diff')
        self.assertEqual(shap_values.shape, (self.X.shape[0], len(self.feature_names), len(self.comparer.class_names)))
        self.assertListEqual(shap_values.feature_names, self.feature_names)
        self.assertListEqual(shap_values.output_names, ['(False, False)', '(False, True)', '(True, False)', '(True, True)'])


class TestExplainersMulticlassTask(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y, self.indiv_class_names, self.feature_names = make_multiclass_classification()
        self.comparer = make_simple_comparer(self.X, self.y, self.feature_names)

    def test_generate_indiv_shap_values_should_succeed(self):
        shap_values = generate_shap_explanations(self.comparer, self.X, explanation_type='indiv')
        self.assertEqual(shap_values.shape, (self.X.shape[0], len(self.feature_names), 2 * len(self.indiv_class_names)))
        self.assertListEqual(shap_values.feature_names, self.feature_names)
        self.assertListEqual(shap_values.output_names, ['A.0', 'A.1', 'A.2', 'B.0', 'B.1', 'B.2'])

    def test_generate_bin_diff_shap_values_should_succeed(self):
        self.assertRaises(MethodUndefinedException,
                          lambda: generate_shap_explanations(self.comparer, self.X, explanation_type='bin_diff'))

    def test_generate_mclass_diff_shap_values_should_succeed(self):
        shap_values = generate_shap_explanations(self.comparer, self.X, explanation_type='mclass_diff')
        self.assertEqual(shap_values.shape, (self.X.shape[0], len(self.feature_names), len(self.comparer.class_names)))
        self.assertListEqual(shap_values.feature_names, self.feature_names)
        self.assertListEqual(shap_values.output_names, ['(0, 0)', '(0, 1)', '(0, 2)', '(1, 0)', '(1, 1)', '(1, 2)',
                                                        '(2, 0)', '(2, 1)', '(2, 2)'])


if __name__ == '__main__':
    unittest.main()
