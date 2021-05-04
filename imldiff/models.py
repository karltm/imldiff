from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import itertools

from sklearn.linear_model import LogisticRegression


class SteppedLogisticRegression(LogisticRegression):
    def decision_function(self, X):
        scores = super(SteppedLogisticRegression, self).decision_function(X)
        return scores.astype(int).astype(float)

    
class TwoClassDiffClassifier(BaseEstimator, ClassifierMixin):
   
    def  __init__(self, clf_a, clf_b):
        self.clf_a = clf_a
        self.clf_b = clf_b

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = [False, True]
        self.clf_a.fit(X, y)
        self.clf_b.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        y_a = self.clf_a.predict(X)
        y_b = self.clf_b.predict(X)
        return y_a == y_b

    def predict_proba(self, X):
        pass

    def predict_log_proba(self, X):
        pass


class NClassDiffClassifier(BaseEstimator, ClassifierMixin):
    
    def  __init__(self, clf_a, clf_b):
        self.clf_a = clf_a
        self.clf_b = clf_b

    def fit(self, X, y):
        X, y = check_X_y(X, y) 
        self.base_classes = unique_labels(y)
        n_target_classes = len(self.base_classes)
        self.classes_ = list(itertools.product(range(n_target_classes), range(n_target_classes)))
        self.clf_a.fit(X, y)
        self.clf_b.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        y_a = self.clf_a.predict(X)
        y_b = self.clf_b.predict(X)
        pass

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        proba_a = self.clf_a.predict_proba(X)
        proba_b = self.clf_a.predict_proba(X)
        pass

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if hasattr(self.clf_a, 'predict_log_proba'):
            log_proba_a = self.clf_a.predict_log_proba(X)
        else:
            proba_a = self.clf_a.predict_proba(X)
            log_proba_a = np.log(proba_a)
        if hasattr(self.clf_b, 'predict_log_proba'):
            log_proba_b = self.clf_b.predict_log_proba(X)
        else:
            proba_b = self.clf_b.predict_proba(X)
            log_proba_b = np.log(proba_b)
        pass


