from scipy.special import logit, expit
import numpy as np


class Classifier:
    pass


class ModelClassifier(Classifier):

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        if hasattr(self.model, 'predict_log_proba'):
            return self.model.predict_log_proba(X)
        else:
            return np.log(self.predict_proba(X))

    def predict_log_odds(self, X):
        log_proba = self.predict_log_proba(X)
        return np.vstack([log_proba[:, 1] - log_proba[:, 0],
                           log_proba[:, 0] - log_proba[:, 1]]).T


class PredictProbabilityMethodClassifier(Classifier):

    def __init__(self, predict_proba_pos):
        self._predict_proba_pos = predict_proba_pos

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(bool)

    def predict_proba(self, X):
        proba_pos = self._predict_proba_pos(X)
        return np.vstack([1-proba_pos, proba_pos]).T

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_log_odds(self, X):
        return logit(self.predict_proba(X))


class PredictLogOddsMethodClassifier(Classifier):

    def __init__(self, predict_log_odds_pos):
        self._predict_log_odds_pos = predict_log_odds_pos

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(bool)

    def predict_proba(self, X):
        return expit(self.predict_log_odds(X))

    def predict_log_proba(self, X):
        return -np.log1p(np.exp(-self.predict_log_odds(X)))

    def predict_log_odds(self, X):
        log_odds_pos = self._predict_log_odds_pos(X)
        return np.vstack([-log_odds_pos, log_odds_pos]).T
    

class MergedClassifier(Classifier):

    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self.classifiers = [self.c1, self.c2]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(bool)

    def predict_log_odds(self, X):
        log_proba = self.predict_log_proba(X)
        log_odds_pos = log_proba[:, 1] - log_proba[:, 0]
        return np.vstack([-log_odds_pos, log_odds_pos]).T


class C1PositiveAndC2Negative(MergedClassifier):

    def predict_proba(self, X):
        proba_pos = self.c1.predict_proba(X)[:, 1] * self.c2.predict_proba(X)[:, 0]
        return np.vstack([1-proba_pos, proba_pos]).T

    def predict_log_proba(self, X):
        log_proba_pos = self.c1.predict_log_proba(X)[:, 1] + self.c2.predict_log_proba(X)[:, 0]
        return np.vstack([log_proba_complement(log_proba_pos), log_proba_pos]).T


def log_proba_complement(log_proba):
    return np.log1p(-np.exp(log_proba))


class C1NegativeAndC2Positive(MergedClassifier):

    def predict_proba(self, X):
        proba_pos = self.c1.predict_proba(X)[:, 0] * self.c2.predict_proba(X)[:, 1]
        return np.vstack([1-proba_pos, proba_pos]).T

    def predict_log_proba(self, X):
        log_proba_pos = self.c1.predict_log_proba(X)[:, 0] + self.c2.predict_log_proba(X)[:, 1]
        return np.vstack([log_proba_complement(log_proba_pos), log_proba_pos]).T


class C1AndC2Different(MergedClassifier):

    def predict_proba(self, X):
        proba_c1 = self.c1.predict_proba(X)
        proba_c2 = self.c2.predict_proba(X)
        proba_pos = proba_c1[:, 1] * proba_c2[:, 0] + proba_c1[:, 0] * proba_c2[:, 1]
        return np.vstack([1-proba_pos, proba_pos]).T

    def predict_log_proba(self, X):
        c1_log_proba = self.c1.predict_log_proba(X)
        c2_log_proba = self.c2.predict_log_proba(X)
        log_proba_pos = log_proba_add(c1_log_proba[:, 1] + c2_log_proba[:, 0],
                                      c1_log_proba[:, 0] + c2_log_proba[:, 1])
        return np.vstack([log_proba_complement(log_proba_pos), log_proba_pos]).T


def log_proba_add(log_proba1, log_proba2):
    return log_proba1 + np.log1p(np.exp(log_proba2 - log_proba1))


class LogOddsRatio(MergedClassifier):

    def predict_log_odds(self, X):
        return self.c1.predict_log_odds(X) - self.c2.predict_log_odds(X)
    