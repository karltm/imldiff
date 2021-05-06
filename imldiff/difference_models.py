from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import itertools
import numpy as np


def predict_log_proba(clf, X):
    if hasattr(clf, 'predict_log_proba'):
        return clf.predict_log_proba(X)
    else:
        proba = clf.predict_proba(X)
        return np.log(proba)
    
def add_in_log_proba_space(log_proba1, log_proba2):
    if np.isfinite(log_proba1) and np.isfinite(log_proba2):
        return log_proba1 + np.log1p(np.exp(log_proba2 - log_proba1))
    if np.isneginf(log_proba1) and np.isneginf(log_proba2):
        return -np.inf
    if np.isneginf(log_proba1) and np.isfinite(log_proba2):
        return log_proba2
    if np.isfinite(log_proba1) and np.isneginf(log_proba2):
        return log_proba1
    return np.nan
    
np_add_in_log_proba_space = np.frompyfunc(add_in_log_proba_space, 2, 1)
complement_log_proba = lambda log_proba: np.log1p(-np.exp(log_proba))
        

class TwoClassDifferenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that classifies whether the passed base classifiers
    predict the same class (True) or a different class (False)
    """
   
    def  __init__(self, clf_a, clf_b, fit_base_classifiers=False):
        self.clf_a = clf_a
        self.clf_b = clf_b
        self.fit_base_classifiers = fit_base_classifiers

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = [False, True]
        if self.fit_base_classifiers:
            self.clf_a.fit(X, y)
            self.clf_b.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels, output shape is (n_samples,)
        """
        check_is_fitted(self)
        X = check_array(X)
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        return pred_a == pred_b

    def predict_proba(self, X):
        """
        Predict probabilities for the two class labels.
        Output shape is (n_samples, 2)
        """
        check_is_fitted(self)
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_b.predict_proba(X)
        proba_pos = np.sum(proba_a * proba_b, axis=1)
        return np.vstack((1-proba_pos, proba_pos)).T

    def predict_log_proba(self, X):
        """
        Predict log-probabilities instead of probabilities
        Output shape is (n_samples, n_labels)
        """
        check_is_fitted(self)
        X = check_array(X)
        log_proba_a = predict_log_proba(self.clf_a, X)
        log_proba_b = predict_log_proba(self.clf_b, X)
        log_proba_pos = np_add_in_log_proba_space.reduce(log_proba_a + log_proba_b, axis=1).astype('float64')
        return np.vstack((complement_log_proba(log_proba_pos), log_proba_pos)).T


class NClassDifferenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that merges the the predictions of the passed base
    classifiers into a flattened version of a confusion matrix
    of the predictions of A and B. The diagonal labels are the cases
    where both classifiers predict the same label, the other where
    they predict differing labels.

    e.g. for a base classification problem consisting of three labels,
    this classifier predicts 9 different labels
         y_B(x) = l1 | l2 | l3
                  ------------
    y_A(x) = l1 | 0    1    2 
    y_A(x) = l2 | 3    4    5
    y_A(x) = l3 | 6    7    8
    """
    
    def  __init__(self, clf_a, clf_b, fit_base_classifiers=False):
        self.clf_a = clf_a
        self.clf_b = clf_b
        self.fit_base_classifiers = fit_base_classifiers

    def fit(self, X, y):
        X, y = check_X_y(X, y) 
        self.base_classes = unique_labels(y)
        self.n_base_classes = len(self.base_classes)
        self.classes_ = np.arange(np.square(self.n_base_classes))
        self.class_tuples = list(itertools.product(self.base_classes, self.base_classes))
        if self.fit_base_classifiers:
            self.clf_a.fit(X, y)
            self.clf_b.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels, output shape is (n_samples,)
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
        Predict probabilities for the n class labels.
        Output shape is (n_samples, n_labels)
        """
        check_is_fitted(self)
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_b.predict_proba(X)
        proba_a_expanded = np.repeat(proba_a, self.n_base_classes, axis=1)
        proba_b_expanded = np.reshape(np.repeat(proba_b, self.n_base_classes, axis=0), proba_a_expanded.shape)
        return proba_a_expanded * proba_b_expanded
        
    def predict_log_proba(self, X):
        """
        Predict log-probabilities instead of probabilities
        Output shape is (n_samples, n_labels)
        """
        check_is_fitted(self)
        X = check_array(X)
        log_proba_a = predict_log_proba(self.clf_a, X)
        log_proba_b = predict_log_proba(self.clf_b, X)
        log_proba_a_expanded = np.repeat(log_proba_a, self.n_base_classes, axis=1)
        log_proba_b_expanded = np.reshape(np.repeat(log_proba_b, self.n_base_classes, axis=0), log_proba_a_expanded.shape)
        return log_proba_a_expanded + log_proba_b_expanded

    
    