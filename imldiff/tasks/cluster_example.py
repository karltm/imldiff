import pandas as pd
import numpy as np
import scipy as sp
from helper_models import RuleClassifier


def decision_rule(X):
    return (X[:, 0] > 0).astype(int)


def modified_decision_rule(X):
    y = decision_rule(X).astype(bool)
    y ^= (-70 < X[:, 0]) & (X[:, 0] <= -60) & (10 < X[:, 1] ) & (X[:, 1] <= 20)    # modification 1
    y ^= (-70 < X[:, 0]) & (X[:, 0] <= -60) & (-20 < X[:, 1] ) & (X[:, 1] <= -10)  # modification 2

    y ^= (-10 < X[:, 0]) & (X[:, 0] <= 0) & (-10 < X[:, 1] ) & (X[:, 1] <= 0)      # modification 3
    y ^= (-20 < X[:, 0]) & (X[:, 0] <= -10) & (-20 < X[:, 1] ) & (X[:, 1] <= -10)  # modification 4
    y ^= (-10 < X[:, 0]) & (X[:, 0] <= 0) & (-30 < X[:, 1] ) & (X[:, 1] <= -20)    # modification 5

    y ^= (0 < X[:, 0]) & (X[:, 0] <= 10) & (20 < X[:, 1] ) & (X[:, 1] <= 30)       # modification 6
    y ^= (10 < X[:, 0]) & (X[:, 0] <= 20) & (10 < X[:, 1] ) & (X[:, 1] <= 20)      # modification 7
    y ^= (0 < X[:, 0]) & (X[:, 0] <= 10) & (0 < X[:, 1] ) & (X[:, 1] <= 10)        # modification 8
    return y.astype(int)


def make_task():
    x1_min, x1_max = -100.0, 100.0
    x2_min, x2_max = -100.0, 100.0
    mesh_step_size = 2.0
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size), np.arange(x2_min, x2_max, mesh_step_size))
    X = np.c_[xx.ravel(), yy.ravel()]

    y = decision_rule(X)

    clf_a = RuleClassifier(decision_rule)
    clf_a.fit(X, y)

    clf_b = RuleClassifier(modified_decision_rule)
    clf_b.fit(X, y)

    feature_names = np.array(['x1', 'x2'])

    return clf_a, clf_b, X, y, feature_names
