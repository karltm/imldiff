import unittest
import numpy as np
import shap
from shap.maskers import Independent, Partition


def make_multiclass_classification():
    class_names = ['c1', 'c2', 'c3']
    f = lambda X: np.repeat(np.repeat(0.0, X.shape[0]).reshape((X.shape[0], 1)), len(class_names), axis=1)
    feature_names = ['x1', 'x2', 'x3', 'x4']
    n_instances = 10
    X = np.repeat(np.arange(n_instances).reshape((n_instances, 1)), len(feature_names), axis=1)
    return X, class_names, f, feature_names


def make_single_output_classification():
    f = lambda X: np.repeat(0.0, X.shape[0])
    feature_names = ['x1', 'x2', 'x3', 'x4']
    n_instances = 10
    X = np.repeat(np.arange(n_instances).reshape((n_instances, 1)), len(feature_names), axis=1)
    return X, f, feature_names


class ExactExplainerTestCase(unittest.TestCase):

    def test_explainer_creation_single_output_model(self):
        X, f, feature_names = make_single_output_classification()
        masker = Independent(X)
        explainer = shap.Explainer(f, masker, algorithm='exact', feature_names=feature_names)
        self.assertEqual(explainer.feature_names, feature_names)

    def test_explainer_creation_multioutput_model_forgets_class_names(self):
        X, class_names, f, feature_names = make_multiclass_classification()
        masker = Independent(X)
        explainer = shap.Explainer(f, masker, algorithm='exact', feature_names=feature_names, output_names=class_names)
        self.assertEqual(explainer.feature_names, feature_names)
        self.assertNotEqual(explainer.output_names, class_names)


class PermutationExplainerTestCase(unittest.TestCase):

    def test_explainer_creation_single_output_model(self):
        X, f, feature_names = make_single_output_classification()
        explainer = self.make_explainer(X, 'class', f, feature_names)
        self.assertEqual(explainer.feature_names, feature_names)
        self.assertEqual(explainer.output_names_backup, 'class')

    def make_explainer(self, X, class_names, f, feature_names):
        masker = Independent(X)
        explainer = shap.Explainer(f, masker, algorithm='permutation', feature_names=feature_names,
                                   output_names=class_names)
        if explainer.output_names is None:
            explainer.output_names_backup = class_names
        return explainer

    def test_explainer_creation_multioutput_model_forgets_class_names(self):
        """ This test demonstrates a bug in shap package.

        see make_multioutput_shap_values for a fix
        """
        X, class_names, f, feature_names = make_multiclass_classification()
        explainer = self.make_explainer(X, class_names, f, feature_names)
        self.assertEqual(explainer.feature_names, feature_names)
        self.assertNotEqual(explainer.output_names, class_names)
        self.assertEqual(explainer.output_names_backup, class_names)

    def test_shap_value_generation_single_output_model_should_succeed(self):
        X, feature_names, instance_names, shap_values = self.make_single_output_shap_values()
        self.assertEqual(shap_values.shape, (X.shape[0], len(feature_names)))
        self.assertEqual(shap_values.feature_names, feature_names)
        self.assertEqual(shap_values.output_names, 'class')
        self.assertEqual(shap_values.instance_names, instance_names)

    def make_single_output_shap_values(self):
        X, f, feature_names = make_single_output_classification()
        instance_names = np.arange(X.shape[0]).astype(str).tolist()
        explainer = self.make_explainer(X, 'class', f, feature_names)
        shap_values = explainer(X)
        if hasattr(explainer, 'output_names_backup'):
            output_names = explainer.output_names_backup
        else:
            output_names = None
        shap_values = shap.Explanation(shap_values.values, shap_values.base_values, shap_values.data,
                                       shap_values.display_data, instance_names,
                                       shap_values.feature_names, output_names)
        return X, feature_names, instance_names, shap_values

    def test_shap_value_generation_multioutput_model_shoud_succeed(self):
        X, class_names, feature_names, instance_names, shap_values = self.make_multioutput_shap_values()
        self.assertEqual(shap_values.shape, (X.shape[0], len(feature_names), len(class_names)))
        self.assertEqual(shap_values.feature_names, feature_names)
        self.assertEqual(shap_values.output_names, class_names)
        self.assertEqual(shap_values.instance_names, instance_names)

    def make_multioutput_shap_values(self):
        """ As a workaround, a new shap.Explanation object is created with the correct class names"""
        X, class_names, f, feature_names = make_multiclass_classification()
        instance_names = np.arange(X.shape[0]).astype(str).tolist()
        explainer = self.make_explainer(X, class_names, f, feature_names)
        shap_values = explainer(X)
        if hasattr(explainer, 'output_names_backup'):
            output_names = explainer.output_names_backup
        else:
            output_names = None

        shap_values = shap.Explanation(shap_values.values, shap_values.base_values, shap_values.data,
                                       shap_values.display_data, instance_names,
                                       shap_values.feature_names, output_names)
        return X, class_names, feature_names, instance_names, shap_values

    def test_slice_instances_single_output_model_shap_values_should_succeed(self):
        X, feature_names, instance_names, shap_values = self.make_single_output_shap_values()
        shap_value_slice = shap_values[2:5]
        self.assertEqual(shap_value_slice.shape, (3, len(feature_names)))
        self.assertEqual(shap_value_slice.instance_names, instance_names[2:5])

    def test_slice_instances_by_mask_single_output_model_shap_values_does_not_work_correctly(self):
        X, feature_names, instance_names, shap_values = self.make_single_output_shap_values()
        mask = np.repeat(False, X.shape[0])
        mask[1] = True
        mask[3] = True
        shap_value_slice = shap_values[mask]
        self.assertEqual(shap_value_slice.shape, (2, len(feature_names)))
        # Because of a bug, we see the following:
        self.assertEqual(shap_value_slice.instance_names, ['0', '1', '0', '1', '0', '0', '0', '0', '0', '0'])
        # But would expect ['1', '3']

    def test_slice_instances_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, instance_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[2:5]
        self.assertEqual(shap_value_slice.shape, (3, len(feature_names), len(class_names)))
        self.assertEqual(shap_value_slice.instance_names, instance_names[2:5])

    def test_slice_features_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, instance_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[:, ['x2', 'x1'], :]
        self.assertEqual(shap_value_slice.shape, (X.shape[0], 2, len(class_names)))
        self.assertEqual(shap_value_slice.feature_names, ['x2', 'x1'])
        self.assertEqual(shap_value_slice.output_names, class_names)
        self.assertEqual(shap_value_slice.instance_names, instance_names)

    def test_slice_single_feature_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, instance_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[:, 'x2', :]
        self.assertEqual(shap_value_slice.shape, (X.shape[0], len(class_names)))
        self.assertEqual(shap_value_slice.feature_names, 'x2')
        self.assertEqual(shap_value_slice.output_names, class_names)
        self.assertEqual(shap_value_slice.instance_names, instance_names)

    def test_slice_classes_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, instance_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[:, :, ['c2', 'c1']]
        self.assertEqual(shap_value_slice.shape, (X.shape[0], len(feature_names), 2))
        self.assertEqual(shap_value_slice.feature_names, feature_names)
        self.assertEqual(shap_value_slice.output_names, ['c2', 'c1'])
        self.assertEqual(shap_value_slice.instance_names, instance_names)

    def test_slice_single_class_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, instance_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[:, :, 'c1']
        self.assertEqual(shap_value_slice.shape, (X.shape[0], len(feature_names)))
        self.assertEqual(shap_value_slice.feature_names, feature_names)
        self.assertEqual(shap_value_slice.output_names, 'c1')
        self.assertEqual(shap_value_slice.instance_names, instance_names)


class PartitionExplainerTestCase(unittest.TestCase):

    def test_explainer_creation_single_output_model(self):
        X, f, feature_names = make_single_output_classification()
        explainer = self.make_explainer(X, None, f, feature_names)
        self.assertEqual(explainer.feature_names, feature_names)
        self.assertEqual(explainer.output_names, None)

    def make_explainer(self, X, class_names, f, feature_names):
        masker = Partition(X)
        explainer = shap.Explainer(f, masker, algorithm='partition', feature_names=feature_names,
                                   output_names=class_names)
        return explainer

    def test_explainer_creation_multioutput_model_should_succeed(self):
        X, class_names, f, feature_names = make_multiclass_classification()
        explainer = self.make_explainer(X, class_names, f, feature_names)
        self.assertEqual(explainer.feature_names, feature_names)
        self.assertEqual(explainer.output_names, class_names)

    def test_shap_value_generation_single_output_model_should_succeed(self):
        X, feature_names, shap_values = self.make_single_output_shap_values()
        self.assertEqual(shap_values.shape, (X.shape[0], len(feature_names)))
        self.assertEqual(shap_values.feature_names, feature_names)
        self.assertEqual(shap_values.output_names, None)

    def make_single_output_shap_values(self):
        X, f, feature_names = make_single_output_classification()
        explainer = self.make_explainer(X, None, f, feature_names)
        shap_values = explainer(X)
        return X, feature_names, shap_values

    def test_shap_value_generation_multioutput_model_shoud_succeed(self):
        X, class_names, feature_names, shap_values = self.make_multioutput_shap_values()
        self.assertEqual(shap_values.shape, (X.shape[0], len(feature_names), len(class_names)))
        self.assertEqual(shap_values.feature_names, feature_names)
        self.assertEqual(shap_values.output_names, class_names)

    def make_multioutput_shap_values(self):
        X, class_names, f, feature_names = make_multiclass_classification()
        explainer = self.make_explainer(X, class_names, f, feature_names)
        shap_values = explainer(X)
        return X, class_names, feature_names, shap_values

    def test_slice_instances_single_output_model_shap_values_should_succeed(self):
        X, feature_names, shap_values = self.make_single_output_shap_values()
        shap_value_slice = shap_values[2:5]
        self.assertEqual(shap_value_slice.shape, (3, len(feature_names)))

    def test_slice_instances_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[2:5]
        self.assertEqual(shap_value_slice.shape, (3, len(feature_names), len(class_names)))

    def test_slice_features_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[:, ['x2', 'x1'], :]
        self.assertEqual(shap_value_slice.shape, (X.shape[0], 2, len(class_names)))
        self.assertEqual(shap_value_slice.feature_names, ['x2', 'x1'])
        self.assertEqual(shap_value_slice.output_names, class_names)

    def test_slice_single_feature_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[:, 'x2', :]
        self.assertEqual(shap_value_slice.shape, (X.shape[0], len(class_names)))
        self.assertEqual(shap_value_slice.feature_names, 'x2')
        self.assertEqual(shap_value_slice.output_names, class_names)

    def test_slice_classes_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[:, :, ['c2', 'c1']]
        self.assertEqual(shap_value_slice.shape, (X.shape[0], len(feature_names), 2))
        self.assertEqual(shap_value_slice.feature_names, feature_names)
        self.assertEqual(shap_value_slice.output_names, ['c2', 'c1'])

    def test_slice_single_class_multioutput_model_shap_values_should_succeed(self):
        X, class_names, feature_names, shap_values = self.make_multioutput_shap_values()
        shap_value_slice = shap_values[:, :, 'c1']
        self.assertEqual(shap_value_slice.shape, (X.shape[0], len(feature_names)))
        self.assertEqual(shap_value_slice.feature_names, feature_names)
        self.assertEqual(shap_value_slice.output_names, 'c1')


if __name__ == '__main__':
    unittest.main()
