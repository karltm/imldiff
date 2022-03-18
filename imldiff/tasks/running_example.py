# Source: https://gitlab.com/andsta/diro2c/-/blob/develop/show_decision_boundaries_running_ex.py
import copy
from sklearn.datasets import make_classification, make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

feature_names = np.array(['x1', 'x2'])
feature_precisions = [1, 1]


def make_task():
    X1, y1 = make_classification(n_samples=300, n_features=2,
                                 n_informative=1, n_redundant=0, n_classes=2, random_state=2, n_clusters_per_class=1, class_sep=1.8, flip_y=0, scale=100)

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=1)
    # -----------------------------------------------
    # train BB 1:

    blackbox1 = DecisionTreeClassifier(random_state=1)
    blackbox1.fit(X1_train, y1_train)

    # -----------------------------------------------
    # manipulate and train BB 2:
    X2 = copy.deepcopy(X1)
    y2 = copy.deepcopy(y1)
    y2_new = []
    for x_0, y_0 in zip(X2, y2):
        if x_0[0] < 150 and x_0[0] > 0 and x_0[1] > -100 and x_0[1] < 100 and y_0 == 1:
            y2_new.append(0)
        elif x_0[0] < 0 and x_0[0] > -200 and x_0[1] >= 100 and y_0 == 0:
            y2_new.append(1)
        else:
            y2_new.append(y_0)

    X2 = np.asarray(X2)
    y2 = np.asarray(y2_new)

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=1)

    blackbox2 = DecisionTreeClassifier(random_state=1)
    blackbox2.fit(X2_train, y2_train)

    return blackbox1, blackbox2, X1, y1, feature_names, feature_precisions


def make_test_set():
    X, _ = make_blobs(n_samples=10000, centers=1, cluster_std=100, random_state=2)
    return X
