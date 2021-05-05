import unittest
from unittest.mock import MagicMock
from difference_models import TwoClassDifferenceClassifier, NClassDifferenceClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


class TestTwoClassDifferenceClassifierOnBinaryTask(unittest.TestCase):
    
    def setUp(self):
        X = np.array([[0, 0], [1, 1]])
        y = np.array([False, True])
        self.clf_a = MagicMock()
        self.clf_b = MagicMock()
        self.diff_clf = TwoClassDifferenceClassifier(self.clf_a, self.clf_b)
        self.diff_clf.fit(X, y)

    def test_predict_equal(self):
        self.clf_a.predict.return_value = np.array([False])
        self.clf_b.predict.return_value = np.array([False])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([True]))

    def test_predict_not_equal(self):
        self.clf_a.predict.return_value = np.array([False])
        self.clf_b.predict.return_value = np.array([True])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([False]))
        
    def test_predict_equal_proba_0(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[0.0, 1.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[1.0, 0.0]]))
        
    def test_predict_proba_equal_100pct(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[1.0, 0.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.0, 1.0]]))
        
    def test_predict_proba_when_both_classifiers_are_uncertain(self):
        self.clf_a.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.clf_b.predict_proba.return_value = np.array([[0.6, 0.4]])
        proba_pos = 0.3*0.6 + 0.7*0.4
        assert_allclose(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[1-proba_pos, proba_pos]]))
        
    def test_predict_log_proba_equal_0pct(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[1.0, 0.0]])))
        
    def test_predict_log_proba_equal_100pct(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.0, 1.0]])))
        
    def test_predict_log_proba_when_both_classifiers_are_uncertain(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.3, 0.7]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.6, 0.4]]))
        proba_pos = 0.3*0.6 + 0.7*0.4
        assert_allclose(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[1-proba_pos, proba_pos]])))
        
        
class TestTwoClassDifferenceClassifierOnTernaryTask(unittest.TestCase):
    
    def setUp(self):
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 1, 2])
        self.clf_a = MagicMock()
        self.clf_b = MagicMock()
        self.diff_clf = TwoClassDifferenceClassifier(self.clf_a, self.clf_b)
        self.diff_clf.fit(X, y)
        
    def test_predict_equal(self):
        self.clf_a.predict.return_value = np.array([1])
        self.clf_b.predict.return_value = np.array([1])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([True]))
        
    def test_predict_not_equal(self):
        self.clf_a.predict.return_value = np.array([1])
        self.clf_b.predict.return_value = np.array([2])
        assert_array_equal(
            self.diff_clf.predict(np.array([[0, 0]])),
            np.array([False]))
        
    def test_predict_proba_equal_0pct(self):
        self.clf_a.predict_proba.return_value = np.array([[1.0, 0.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[0.0, 1.0, 0.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[1.0, 0.0]]))
        
    def test_predict_proba_equal_100pct(self):
        self.clf_a.predict_proba.return_value = np.array([[0.0, 1.0, 0.0]])
        self.clf_b.predict_proba.return_value = np.array([[0.0, 1.0, 0.0]])
        assert_array_equal(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[0.0, 1.0]]))
        
    def test_predict_proba_when_both_classifiers_are_uncertain(self):
        self.clf_a.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        self.clf_b.predict_proba.return_value = np.array([[0.3, 0.2, 0.5]])
        proba_pos = 0.1*0.3 + 0.2*0.2 + 0.7*0.5
        assert_allclose(
            self.diff_clf.predict_proba(np.array([[0, 0]])),
            np.array([[1-proba_pos, proba_pos]]))
        
    def test_predict_log_proba_equal_0pct(self):
        np.seterr(divide='ignore', invalid='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[1.0, 0.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0, 0.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[1.0, 0.0]])))
        
    def test_predict_log_proba_equal_100pct(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0, 0.0]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.0, 1.0, 0.0]]))
        assert_array_equal(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[0.0, 1.0]])))
        
    def test_predict_log_proba_when_both_classifiers_are_uncertain(self):
        np.seterr(divide='ignore') 
        self.clf_a.predict_log_proba.return_value = np.log(np.array([[0.1, 0.2, 0.7]]))
        self.clf_b.predict_log_proba.return_value = np.log(np.array([[0.3, 0.2, 0.5]]))
        proba_pos = 0.1*0.3 + 0.2*0.2 + 0.7*0.5
        assert_allclose(
            self.diff_clf.predict_log_proba(np.array([[0, 0]])),
            np.log(np.array([[1-proba_pos, proba_pos]])))
        
        
class TestNClassDifferenceClassifierOnBinaryTask(unittest.TestCase):
    
    def setUp(self):
        X = np.array([[0, 0], [1, 1]])
        y = np.array([False, True])
        self.clf_a = MagicMock()
        self.clf_b = MagicMock()
        self.diff_clf = NClassDifferenceClassifier(self.clf_a, self.clf_b)
        self.diff_clf.fit(X, y)
        
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
        
        
class TestNClassDifferenceClassifierOnTernaryTask(unittest.TestCase):
    
    def setUp(self):
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 1, 2])
        self.clf_a = MagicMock()
        self.clf_b = MagicMock()
        self.diff_clf = NClassDifferenceClassifier(self.clf_a, self.clf_b)
        self.diff_clf.fit(X, y)
        
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