import unittest
from unittest.mock import MagicMock
from difference_models import BinaryDifferenceClassifier, MulticlassDifferenceClassifier, MethodUndefinedException
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


class BaseClassifier(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        return self

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def predict_log_proba(self, X):
        pass


class TestBinaryDifferenceClassifierOnBinaryTask(unittest.TestCase):
    X = np.array([[0, 0], [1, 1]])
    y = np.array([False, True])
    
    def setUp(self):
        self.clf_a = self.make_clf()
        self.clf_b = self.make_clf()
        self.diff_clf = BinaryDifferenceClassifier(self.clf_a, self.clf_b)

    def make_clf(self):
        clf = MagicMock(spec=BaseClassifier())
        clf.classes_ = np.unique(self.y)
        return clf

    def test_predict_not_different(self):
        self.clf_a.predict.return_value = np.array([False])
        self.clf_b.predict.return_value = np.array([False])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([False]))

    def test_predict_different(self):
        self.clf_a.predict.return_value = np.array([False])
        self.clf_b.predict.return_value = np.array([True])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([True]))
        
    def test_predict_proba_different_100pct(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[0.0, 1.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.0, 1.0]]))
        
    def test_predict_proba_different_0pct(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[1.0, 0.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[1.0, 0.0]]))
        
    def test_predict_proba_when_both_classifiers_are_uncertain(self):
        self.clf_a.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.clf_b.predict_proba.return_value = np.array([[0.6, 0.4]])
        proba_equal = 0.3*0.6 + 0.7*0.4
        assert_allclose(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[proba_equal, 1-proba_equal]]))
        
    def test_predict_log_proba_different_100pct(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.0, 1.0]])))
        
    def test_predict_log_proba_different_0pct(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[1.0, 0.0]])))
        
    def test_predict_log_proba_when_both_classifiers_are_uncertain(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.3, 0.7]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.6, 0.4]]))
        proba_equal = 0.3*0.6 + 0.7*0.4
        assert_allclose(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[proba_equal, 1-proba_equal]])))
        
        
class TestBinaryDifferenceClassifierOnTernaryTask(unittest.TestCase):
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([0, 1, 2])

    def setUp(self):
        self.clf_a = self.make_clf()
        self.clf_b = self.make_clf()
        self.diff_clf = BinaryDifferenceClassifier(self.clf_a, self.clf_b)

    def make_clf(self):
        clf = MagicMock(spec=BaseClassifier())
        clf.classes_ = np.unique(self.y)
        return clf
        
    def test_predict_not_different(self):
        self.clf_a.predict.return_value = np.array([1])
        self.clf_b.predict.return_value = np.array([1])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([False]))
        
    def test_predict_different(self):
        self.clf_a.predict.return_value = np.array([1])
        self.clf_b.predict.return_value = np.array([2])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([True]))
        
    def test_predict_proba_raises_exception(self):
        self.clf_a.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        self.clf_b.predict_proba.return_value = np.array([[0.3, 0.2, 0.5]])
        self.assertRaises(MethodUndefinedException, lambda: self.diff_clf.predict_log_proba(np.array([[0, 0]])))
        
    def test_predict_log_proba_raises_exception(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0, 0.0]]))
        self.assertRaises(MethodUndefinedException, lambda: self.diff_clf.predict_log_proba(np.array([[0, 0]])))
        
        
class TestMulticlassDifferenceClassifierOnBinaryTask(unittest.TestCase):
    X = np.array([[0, 0], [1, 1]])
    y = np.array([False, True])

    def setUp(self):
        self.clf_a = self.make_clf()
        self.clf_b = self.make_clf()
        self.diff_clf = MulticlassDifferenceClassifier(self.clf_a, self.clf_b)

    def make_clf(self):
        clf = MagicMock(spec=BaseClassifier())
        clf.classes_ = np.unique(self.y)
        return clf
        
    def test_predict_false_false(self):
        self.clf_a.predict.return_value = np.array([False])
        self.clf_b.predict.return_value = np.array([False])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([0]))
        
    def test_predict_false_true(self):
        self.clf_a.predict.return_value = np.array([False])
        self.clf_b.predict.return_value = np.array([True])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([1]))
        
    def test_predict_true_false(self):
        self.clf_a.predict.return_value = np.array([True])
        self.clf_b.predict.return_value = np.array([False])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([2]))
        
    def test_predict_true_true(self):
        self.clf_a.predict.return_value = np.array([True])
        self.clf_b.predict.return_value = np.array([True])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([3]))
        
    def test_predict_proba_when_both_predict_negative_class_100pct(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[1.0, 0.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[1.0, 0.0, 0.0, 0.0]]))
        
    def test_predict_proba_when_a_predicts_negative_class_100pct_b_predicts_positive_class_100pct(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[0.0, 1.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.0, 1.0, 0.0, 0.0]]))
        
    def test_predict_proba_when_a_predicts_positive_class_100pct_b_predicts_negative_class_100pct(self):
        self.clf_a.predict_proba.return_value = np.array([[0.0, 1.0]])
        self.clf_b.predict_proba.return_value = np.array([[1.0, 0.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.0, 0.0, 1.0, 0.0]]))
        
    def test_predict_proba_when_both_predict_positive_class_100pct(self):
        self.clf_a.predict_proba.return_value = np.array([[0.0, 1.0]])
        self.clf_b.predict_proba.return_value = np.array([[0.0, 1.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.0, 0.0, 0.0, 1.0]]))
        
    def test_predict_proba_when_both_classifiers_are_uncertain(self):
        self.clf_a.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.clf_b.predict_proba.return_value = np.array([[0.6, 0.4]])
        assert_allclose(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.3*0.6, 0.3*0.4, 0.7*0.6, 0.7*0.4]]))

    def test_predict_log_proba_when_both_predict_negative_class_100pct(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[1.0, 0.0, 0.0, 0.0]])))
        
    def test_predict_log_proba_when_a_predicts_negative_class_100pct_b_predicts_positive_class_100pct(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.0, 1.0, 0.0, 0.0]])))
        
    def test_predict_log_proba_when_a_predicts_positive_class_100pct_b_predicts_negative_class_100pct(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.0, 0.0, 1.0, 0.0]])))
        
    def test_predict_log_proba_when_both_predict_positive_class_100pct(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.0, 0.0, 0.0, 1.0]])))
        
    def test_predict_log_proba_when_both_classifiers_are_uncertain(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.3, 0.7]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.6, 0.4]]))
        assert_allclose(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.3*0.6, 0.3*0.4, 0.7*0.6, 0.7*0.4]])))
        
        
class TestMulticlassDifferenceClassifierOnTernaryTask(unittest.TestCase):
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([0, 1, 2])

    def setUp(self):
        self.clf_a = self.make_clf()
        self.clf_b = self.make_clf()
        self.diff_clf = MulticlassDifferenceClassifier(self.clf_a, self.clf_b)

    def make_clf(self):
        clf = MagicMock(spec=BaseClassifier())
        clf.classes_ = np.unique(self.y)
        return clf
        
    def test_predict_when_both_predict_label1(self):
        self.clf_a.predict.return_value = np.array([[0]])
        self.clf_b.predict.return_value = np.array([[0]])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([[0]]))
        
    def test_predict_when_a_predicts_label1_b_predicts_label3(self):
        self.clf_a.predict.return_value = np.array([[0]])
        self.clf_b.predict.return_value = np.array([[2]])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([[2]]))
        
    def test_predict_when_a_predicts_label3_b_predicts_label1(self):
        self.clf_a.predict.return_value = np.array([[2]])
        self.clf_b.predict.return_value = np.array([[0]])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([[6]]))
        
    def test_predict_proba_when_both_predict_label1(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[1.0, 0.0, 0.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
        
    def test_predict_proba_when_a_predicts_label1_b_predicts_label3(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[0.0, 0.0, 1.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
        
    def test_predict_proba_when_a_predicts_label3_b_predicts_label1(self):
        self.clf_a.predict_proba.return_value = np.array([[0.0, 0.0, 1.0]])
        self.clf_b.predict_proba.return_value = np.array([[1.0, 0.0, 0.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))
        
    def test_predict_proba_when_both_classifiers_are_uncertain(self):
        self.clf_a.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        self.clf_b.predict_proba.return_value = np.array([[0.3, 0.2, 0.5]])
        assert_allclose(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.1*0.3, 0.1*0.2, 0.1*0.5,
                       0.2*0.3, 0.2*0.2, 0.2*0.5,
                       0.7*0.3, 0.7*0.2, 0.7*0.5]]))
        
    def test_predict_log_proba_when_both_predict_label1(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0, 0.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])))
        
    def test_predict_log_proba_when_a_predicts_label1_b_predicts_label3(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.0, 0.0, 1.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])))
        
    def test_predict_log_proba_when_a_predicts_label3_b_predicts_label1(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.0, 0.0, 1.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0, 0.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])))
        
    def test_predict_log_proba_when_both_classifiers_are_uncertain(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.1, 0.2, 0.7]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.3, 0.2, 0.5]]))
        assert_allclose(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.1*0.3, 0.1*0.2, 0.1*0.5,
                              0.2*0.3, 0.2*0.2, 0.2*0.5,
                              0.7*0.3, 0.7*0.2, 0.7*0.5]])))
        

if __name__ == '__main__':
    unittest.main()
