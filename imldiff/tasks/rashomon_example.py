import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def make_task():
    X_train1, y = make_classification(n_samples=300, n_features=3, n_redundant=0, n_informative=2,
                                      n_clusters_per_class=1, class_sep=0.75, random_state=0)
    X_train1 *= 100
    feature_names = np.array(['x1', 'x2', 'x3'])

    X = X_train1.copy()
    X[:, 0] = X[:, 1]

    lm = LogisticRegression()
    lm.fit(X_train1, y)
    clf_a = lm

    clf_b = LogisticRegression()
    X_train2 = X_train1.copy()
    X_train2[:, 0] = X_train1[:, 1]
    X_train2[:, 1] = X_train1[:, 0]
    clf_b.fit(X_train2, y)

    return clf_a, clf_b, X, y, feature_names
