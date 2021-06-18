import unittest
import numpy as np
from comparers import ModelComparer
from sklearn.dummy import DummyClassifier
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
        self.X, self.y, self.class_names, self.feature_names = make_binary_classification()
        self.comparer = make_simple_comparer(self.X, self.y, self.feature_names)

    def test_generate_indiv_shap_values_should_succeed(self):
        explanations = generate_shap_explanations(self.comparer, self.X, explanation_types=['indiv', 'indiv_diff'])

        self.assertEqual(explanations.indiv.labels.A.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.labels.B.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.labels.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.indiv.proba.A.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.proba.B.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.proba.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.indiv.log_odds.A.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.log_odds.B.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.log_odds.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.indiv.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.indiv_diff.labels.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv_diff.proba.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv_diff.log_odds.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv_diff.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.shape, (self.X.shape[0], len(self.feature_names)))

    def test_generate_bindiff_shap_values_should_succeed(self):
        explanations = generate_shap_explanations(self.comparer, self.X, explanation_types=['bin_diff'])
        self.assertEqual(explanations.bin_diff.labels.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertEqual(explanations.bin_diff.proba.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertEqual(explanations.bin_diff.log_odds.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertEqual(explanations.bin_diff.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertEqual(explanations.shape, (self.X.shape[0], len(self.feature_names)))

    def test_generate_mclass_diff_shap_values_should_succeed(self):
        explanations = generate_shap_explanations(self.comparer, self.X, explanation_types=['mclass_diff'])
        self.assertEqual(explanations.mclass_diff.labels.shape, (self.X.shape[0], len(self.feature_names), np.square(len(self.class_names))))
        self.assertEqual(explanations.mclass_diff.proba.shape, (self.X.shape[0], len(self.feature_names), np.square(len(self.class_names))))
        self.assertEqual(explanations.mclass_diff.log_odds.shape, (self.X.shape[0], len(self.feature_names), np.square(len(self.class_names))))
        self.assertEqual(explanations.mclass_diff.shape, (self.X.shape[0], len(self.feature_names), np.square(len(self.class_names))))
        self.assertEqual(explanations.shape, (self.X.shape[0], len(self.feature_names)))


class TestExplainersMulticlassTask(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y, self.class_names, self.feature_names = make_multiclass_classification()
        self.comparer = make_simple_comparer(self.X, self.y, self.feature_names)

    def test_generate_indiv_shap_values_should_succeed(self):
        explanations = generate_shap_explanations(self.comparer, self.X, explanation_types=['indiv', 'indiv_diff'])

        self.assertEqual(explanations.indiv.labels.A.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.labels.B.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.labels.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.indiv.proba.A.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.proba.B.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.proba.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.indiv.log_odds.A.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.log_odds.B.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv.log_odds.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.indiv.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.indiv_diff.labels.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv_diff.proba.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv_diff.log_odds.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))
        self.assertEqual(explanations.indiv_diff.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)))

        self.assertEqual(explanations.shape, (self.X.shape[0], len(self.feature_names)))

    def test_generate_bindiff_shap_values_should_succeed(self):
        explanations = generate_shap_explanations(self.comparer, self.X, explanation_types=['bin_diff'])
        self.assertEqual(explanations.bin_diff.labels.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertEqual(explanations.bin_diff.proba.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertEqual(explanations.bin_diff.log_odds.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertEqual(explanations.bin_diff.shape, (self.X.shape[0], len(self.feature_names)))
        self.assertEqual(explanations.shape, (self.X.shape[0], len(self.feature_names)))

    def test_generate_mclass_diff_shap_values_should_succeed(self):
        explanations = generate_shap_explanations(self.comparer, self.X, explanation_types=['mclass_diff'])
        self.assertEqual(explanations.mclass_diff.labels.shape, (self.X.shape[0], len(self.feature_names), np.square(len(self.class_names))))
        self.assertEqual(explanations.mclass_diff.proba.shape, (self.X.shape[0], len(self.feature_names), np.square(len(self.class_names))))
        self.assertEqual(explanations.mclass_diff.log_odds.shape, (self.X.shape[0], len(self.feature_names), np.square(len(self.class_names))))
        self.assertEqual(explanations.mclass_diff.shape, (self.X.shape[0], len(self.feature_names), np.square(len(self.class_names))))
        self.assertEqual(explanations.shape, (self.X.shape[0], len(self.feature_names)))

    def test_merge_indiv_shap_values_should_succeed(self):
        explanations = generate_shap_explanations(self.comparer, self.X, explanation_types=['indiv'])
        merged = explanations.indiv.labels.merged
        self.assertEqual(merged.shape, (self.X.shape[0], len(self.feature_names), len(self.class_names)*2))

    def test_merge_all_shap_values_should_succeed(self):
        explanations = generate_shap_explanations(self.comparer, self.X, explanation_types=['indiv', 'indiv_diff', 'bin_diff', 'mclass_diff'])
        merged = explanations.merged
        n_classes = 3*len(self.class_names)*2 +\
                    3*len(self.class_names) +\
                    3*1 +\
                    3*np.square(len(self.class_names))
        self.assertEqual(merged.shape, (self.X.shape[0], len(self.feature_names), n_classes))


if __name__ == '__main__':
    unittest.main()
