import numpy as np
from helper_models import RuleClassifier


def decision_rule(X):
    return (X[:, 0] > 0).astype(int)


def modified_decision_rule(X):
    y = decision_rule(X).astype(bool)

    mod1 = (-10 < X[:, 0]) & (X[:, 0] <= 0) & (-20 < X[:, 1] ) & (X[:, 1] <= 10)
    mod2 = (-20 < X[:, 0]) & (X[:, 0] <= -10) & (-10 < X[:, 1] ) & (X[:, 1] <= 0)
    mod3 = (-60 < X[:, 0]) & (X[:, 0] <= -50) & (0 < X[:, 1] ) & (X[:, 1] <= 10)
    mod4 = (0 < X[:, 0]) & (X[:, 0] <= 10) & (0 < X[:, 1] ) & (X[:, 1] <= 10)

    y ^= (mod1 | mod2 | mod3 | mod4)

    return y.astype(int)


def make_task():
    x1_min, x1_max = -80.0, 80.0
    x2_min, x2_max = -80.0, 80.0
    mesh_step_size = 5.0
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size), np.arange(x2_min, x2_max, mesh_step_size))
    X = np.c_[xx.ravel(), yy.ravel()]

    y = decision_rule(X)

    clf_a = RuleClassifier(decision_rule)
    clf_a.fit(X, y)

    clf_b = RuleClassifier(modified_decision_rule)
    clf_b.fit(X, y)

    feature_names = np.array(['x1', 'x2'])

    return clf_a, clf_b, X, y, feature_names
