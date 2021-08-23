from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import itertools
import numpy as np
from scipy.special import logsumexp


complement_log_proba = lambda log_proba: np.log1p(-np.exp(log_proba))


class BinaryDifferenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that classifies whether the passed base classifiers
    predict the same class (False) or a different class (True)
    """
   
    def  __init__(self, clf_a, clf_b):
        check_is_fitted(clf_a)
        check_is_fitted(clf_b)
        if not np.array_equal(clf_a.classes_, clf_b.classes_):
            raise Exception('Classes of base classifiers need to be the same')
        self.clf_a = clf_a
        self.clf_b = clf_b
        self.classes_ = np.array([False, True])

    def fit(self, X, y):
        return self

    def predict(self, X):
        """
        Predict class labels, output shape is (n,)
        """
        X = check_array(X)
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        return pred_a != pred_b

    def predict_proba(self, X):
        """
        Predict probabilities for the two classes, the output shape is (n, 2)
        """
        if not len(self.clf_a.classes_) == 2:
            raise Exception('Probability estimates are only available for binary base classifiers')
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_b.predict_proba(X)
        proba_pos = np.sum(proba_a * proba_b, axis=1)
        return np.vstack((proba_pos, 1-proba_pos)).T

    def predict_log_proba(self, X):
        """
        Predict log-probabilities instead of probabilities for the two classes,
        the output shape is (n, 2)
        """
        if not len(self.clf_a.classes_) == 2:
            raise Exception('Probability estimates are only available for binary base classifiers')
        X = check_array(X)
        log_proba_a = self.clf_a.predict_log_proba(X)
        log_proba_b = self.clf_b.predict_log_proba(X)
        log_proba_pos = logsumexp(log_proba_a + log_proba_b, axis=1)
        return np.vstack((log_proba_pos, complement_log_proba(log_proba_pos))).T


class MulticlassDifferenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Reformulates the classification problem to classify into m^2 difference classes,
    m being the number of base classes in the base classification problem.
    
    They can be visualized in a m x m matrix with the corresponding class labels l_1, l_2, ... l_m,
    where the diagonal contains classes where the base classifiers predict equal classes:
    clf_b(x) = l_1     l_2     ...     l_m
    clf_a(x) = ---------------------------
      l_1    |  0       1      ...      m
      l_2    | m+1     m+2     ...     2*m
      ...    |
      l_m    | m^2-m  m^2-m+1  ...     m^2
      
    For binary base classification problems this results in a 4-class difference classifier:
    clf_b(x) = False True
    clf_a(x) = ----------
      False  |  0     1
      True   |  2     3
      
    And for 3-class base classification problems this results in a 9-class difference classifier:
    clf_b(x) = l_1 l_2 l_3
    clf_a(x) = -----------
      l_1    |  0   1   2
      l_2    |  3   4   5
      l_3    |  6   7   8

    After fitting, the following variables are available:
    - `base_classes_`: numpy array of the base classes [l_1, l_2, ... l_m]
    - `classes_`: numpy array of the equality and difference classes [0, 1, ... (m-1)^2]
    - `equality_classes_`: numpy array of the equality classes,
                           the diagonal elements of `classes_`, if reshaped as a m x m matrix
    - `difference_classes_`: numpy array of the difference classes
    - `class_tuples_: numpy array of difference class tuples [(l_1, l_1), (l_1, l_2), ... (l_m, l_m)]
    """
    
    def  __init__(self, clf_a, clf_b):
        check_is_fitted(clf_a)
        check_is_fitted(clf_b)
        if not np.array_equal(clf_a.classes_, clf_b.classes_):
            raise Exception('Classes of base classifiers need to be the same')
        self.clf_a = clf_a
        self.clf_b = clf_b
        self.base_classes_ = clf_a.classes_
        self.classes_ = np.arange(np.square(len(self.base_classes_)))
        self.equality_classes_ = np.diagonal(self.classes_.reshape((len(self.base_classes_), len(self.base_classes_))))
        self.difference_classes_ = np.setdiff1d(self.classes_, self.equality_classes_)
        self.class_tuples_ = np.array(list(itertools.product(self.base_classes_, self.base_classes_)), dtype='object,object').astype(object)

    def fit(self, X, y):
        return self

    def predict(self, X):
        """
        Predict difference class labels, the output shape is (n,).
        """
        X = check_array(X)
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        pred_a_idx = np.searchsorted(self.base_classes_, pred_a)
        pred_b_idx = np.searchsorted(self.base_classes_, pred_b)
        return pred_a_idx * len(self.base_classes_) + pred_b_idx

    def predict_proba(self, X):
        """
        Predict probabilities for the difference classes, the output shape is (n, m)
        """
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_b.predict_proba(X)
        proba_a_expanded = np.repeat(proba_a, len(self.base_classes_), axis=1)
        proba_b_expanded = np.reshape(np.repeat(proba_b, len(self.base_classes_), axis=0), proba_a_expanded.shape)
        return proba_a_expanded * proba_b_expanded
        
    def predict_log_proba(self, X):
        """
        Predict log-probabilities instead of probabilities for the difference classes,
        the output shape is (n, m)
        """
        X = check_array(X)
        log_proba_a = self.clf_a.predict_log_proba(X)
        log_proba_b = self.clf_b.predict_log_proba(X)
        log_proba_a_expanded = np.repeat(log_proba_a, len(self.base_classes_), axis=1)
        log_proba_b_expanded = np.reshape(np.repeat(log_proba_b, len(self.base_classes_), axis=0), log_proba_a_expanded.shape)
        return log_proba_a_expanded + log_proba_b_expanded
