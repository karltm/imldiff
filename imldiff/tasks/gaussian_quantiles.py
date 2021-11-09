import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import make_gaussian_quantiles


def make_task():
    """
    Based on https://gitlab.com/andsta/diro2c/-/blob/develop/show_decision_boundaries_gaussians.py
    Originally published as https://doi.org/10.5281/zenodo.5362220
    Modification: added `probability=True` parameter to SVC instantiation, to enable
                  `predict_proba` and `predict_log_proba` functions of the classifiers
    """
    X1, y1 = make_gaussian_quantiles(n_samples=300,
                                     n_classes=2, shuffle=False, cov=0.8, random_state=7)
    X1 = X1 * 100
    blackbox1 = svm.SVC(random_state=1, probability=True)
    blackbox1.fit(X1, y1)

    X2, y2 = make_gaussian_quantiles(n_samples=300,
                                     n_classes=2, shuffle=False, cov=1.3, random_state=7)
    X2 = X2 * 100
    blackbox2 = svm.SVC(random_state=1, probability=True)
    blackbox2.fit(X2, y2)

    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))

    feature_names = np.array(['x1', 'x2'])

    return blackbox1, blackbox2, X, y, feature_names
