import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def make_task():
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, class_sep=0.75, random_state=0)
    X *= 100
    feature_names = np.array(['x1', 'x2'])
    lm = LogisticRegression()
    lm.fit(X, y)
    clf_a = lm
    clf_b = LogisticRegression()
    X2 = X.copy()
    X2[:, 0] = 0
    clf_b.fit(X2, y)
    return clf_a, clf_b, X, y, feature_names
