from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import itertools
import numpy as np
from scipy.special import logsumexp


class MethodUndefinedException(Exception):
    pass


class BinaryDifferenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that classifies whether the passed base classifiers
    predict a different class label (True) or same class label (False)
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
        Predict class labels
        :param X: input data in array of shape (n, p)
        :return:  predictions in array of shape (n, )
        """
        X = check_array(X)
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        return pred_a != pred_b

    def predict_proba(self, X):
        """
        Predict probabilities for the two class labels
        :param X: input data in array of shape (n, p)
        :return: predictions in array of shape (n, 2)
        """
        if not len(self.clf_a.classes_) == 2:
            raise MethodUndefinedException('Probability estimates are undefined for multiclass classification problems')
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_b.predict_proba(X)
        proba_pos = np.sum(proba_a * proba_b, axis=1)
        return np.vstack((proba_pos, 1-proba_pos)).T

    def predict_log_proba(self, X):
        """
        Predict log-transformed probabilities for the two class labels
        :return: predictions in array of shape (n, 2)
        """
        if not len(self.clf_a.classes_) == 2:
            raise MethodUndefinedException('Probability estimates are undefined for multiclass classification problems')
        X = check_array(X)
        log_proba_a = self.clf_a.predict_log_proba(X)
        log_proba_b = self.clf_b.predict_log_proba(X)
        log_proba_pos = logsumexp(log_proba_a + log_proba_b, axis=1)
        return np.vstack((log_proba_pos, complement_log_proba(log_proba_pos))).T


def complement_log_proba(log_proba):
    return np.log1p(-np.exp(log_proba))


class MulticlassDifferenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Reformulates the classification problem to classify into q^2 difference classes,
    q being the number of base classes in the base classification problem.
    
    They can be visualized in a qxq matrix with the corresponding class labels 0, 1, ... q^2
    where the diagonal contains classes where the base classifiers predict equal classes:
    clf_b(x) = 0        1        ...     q
    clf_a(x)
      =      ---------------------------
      0      |  0       1        ...     q
      1      | q+1      q+2      ...     2q
      ...    |
      q      | q^2-q    q^2-q+1  ...     q^2
      
    For binary base classification problems this results in a 4-class difference classifier:
    clf_b(x) =  0   1
    clf_a(x)
      =      ----------
      0      |  0   1
      1      |  2   3
      
    And for 3-class base classification problems this results in a 9-class difference classifier:
    clf_b(x) = 0   1   2
    clf_a(x)
       =     -----------
       0    |  0   1   2
       1    |  3   4   5
       2    |  6   7   8

    The following variables are available:
    - `base_classes_`: numpy array of the individual classifiers' classes [0, 1, ... q]
    - `classes_`: numpy array of the difference classifier classes [0, 1, ... (q-1)^2]
    - `equality_classes_`: numpy array of the equality classes
    - `difference_classes_`: numpy array of the difference classes
    - `class_tuples_: numpy array of the difference classifier classes as class pairs [(0, 0), (0, 1), ... (q, q)]
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
        Predict class labels
        :param X: input data in array of shape (n, p)
        :return:  predictions in array of shape (n, )
        """
        X = check_array(X)
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        pred_a_idx = np.searchsorted(self.base_classes_, pred_a)
        pred_b_idx = np.searchsorted(self.base_classes_, pred_b)
        return pred_a_idx * len(self.base_classes_) + pred_b_idx

    def predict_proba(self, X):
        """
        Predict probabilities for the q class labels
        :param X: input data in array of shape (n, p)
        :return: predictions in array of shape (n, q)
        """
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_b.predict_proba(X)
        proba_a_expanded = np.repeat(proba_a, len(self.base_classes_), axis=1)
        proba_b_expanded = np.reshape(np.repeat(proba_b, len(self.base_classes_), axis=0), proba_a_expanded.shape)
        return proba_a_expanded * proba_b_expanded
        
    def predict_log_proba(self, X):
        """
        Predict log-transformed probabilities for the q class labels
        :param X: input data in array of shape (n, p)
        :return: predictions in array of shape (n, q)
        """
        X = check_array(X)
        log_proba_a = self.clf_a.predict_log_proba(X)
        log_proba_b = self.clf_b.predict_log_proba(X)
        log_proba_a_expanded = np.repeat(log_proba_a, len(self.base_classes_), axis=1)
        log_proba_b_expanded = np.reshape(np.repeat(log_proba_b, len(self.base_classes_), axis=0), log_proba_a_expanded.shape)
        return log_proba_a_expanded + log_proba_b_expanded
