from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import itertools
import numpy as np
from scipy.special import logsumexp


def predict_log_proba(clf, X):
    if hasattr(clf, 'predict_log_proba'):
        return clf.predict_log_proba(X)
    else:
        proba = clf.predict_proba(X)
        return np.log(proba)

complement_log_proba = lambda log_proba: np.log1p(-np.exp(log_proba))


class BinaryDifferenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that classifies whether the passed base classifiers
    predict the same class (True) or a different class (False)
    """
   
    def  __init__(self, clf_a, clf_b):
        self.clf_a = clf_a
        self.clf_b = clf_b

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.array([False, True])
        return self

    def predict(self, X):
        """
        Predict class labels, output shape is (n,)
        """
        check_is_fitted(self)
        X = check_array(X)
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        return pred_a == pred_b

    def predict_proba(self, X):
        """
        Predict probabilities for the two classes, the output shape is (n, 2)
        """
        check_is_fitted(self)
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_b.predict_proba(X)
        proba_pos = np.sum(proba_a * proba_b, axis=1)
        return np.vstack((1-proba_pos, proba_pos)).T

    def predict_log_proba(self, X):
        """
        Predict log-probabilities instead of probabilities for the two classes,
        the output shape is (n, 2)
        """
        check_is_fitted(self)
        X = check_array(X)
        log_proba_a = predict_log_proba(self.clf_a, X)
        log_proba_b = predict_log_proba(self.clf_b, X)
        log_proba_pos = logsumexp(log_proba_a + log_proba_b, axis=1)
        return np.vstack((complement_log_proba(log_proba_pos), log_proba_pos)).T


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

    Initialize with already fitted base classifiers. After fitting the difference classifier,
    the following variables are available:
    - `base_classes`: numpy array of the base classes [l_1, l_2, ... l_m]
    - `classes_`: numpy array of the difference classes [0, 1, ... (m-1)^2]
    - `class_tuples: list of difference class tuples [(l_1, l_1), (l_1, l_2), ... (l_m, l_m)]
    """
    
    def  __init__(self, clf_a, clf_b):
        self.clf_a = clf_a
        self.clf_b = clf_b

    def fit(self, X, y):
        X, y = check_X_y(X, y) 
        self.base_classes = unique_labels(y)
        self.classes_ = np.arange(np.square(len(self.base_classes)))
        self.class_tuples = list(itertools.product(self.base_classes, self.base_classes))
        return self

    def predict(self, X):
        """
        Predict difference class labels, the output shape is (n,).
        """
        check_is_fitted(self)
        X = check_array(X)
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        pred_a_idx = np.searchsorted(self.base_classes, pred_a)
        pred_b_idx = np.searchsorted(self.base_classes, pred_b)
        return pred_a_idx * len(self.base_classes) + pred_b_idx

    def predict_proba(self, X):
        """
        Predict probabilities for the difference classes, the output shape is (n, m)
        """
        check_is_fitted(self)
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_b.predict_proba(X)
        proba_a_expanded = np.repeat(proba_a, len(self.base_classes), axis=1)
        proba_b_expanded = np.reshape(np.repeat(proba_b, len(self.base_classes), axis=0), proba_a_expanded.shape)
        return proba_a_expanded * proba_b_expanded
        
    def predict_log_proba(self, X):
        """
        Predict log-probabilities instead of probabilities for the difference classes,
        the output shape is (n, m)
        """
        check_is_fitted(self)
        X = check_array(X)
        log_proba_a = predict_log_proba(self.clf_a, X)
        log_proba_b = predict_log_proba(self.clf_b, X)
        log_proba_a_expanded = np.repeat(log_proba_a, len(self.base_classes), axis=1)
        log_proba_b_expanded = np.reshape(np.repeat(log_proba_b, len(self.base_classes), axis=0), log_proba_a_expanded.shape)
        return log_proba_a_expanded + log_proba_b_expanded

    
    