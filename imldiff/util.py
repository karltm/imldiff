from ast import literal_eval

import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.tree import _tree
import numpy as np


def calibrate_classifier(est, name, X_train, X_test, y_train, y_test, cv=10, fig_index=1):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=cv, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=cv, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic (Baseline)'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=max(y_train.max(), y_test.max()))
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

    return isotonic, sigmoid


def get_rules(tree, feature_names):
    """ source: https://mljar.com/blog/extract-rules-decision-tree/ """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = ""

        for p in path[:-1]:
            if rule != "":
                rule += " & "
            rule += str(p)
        rule += f" [{path[-1][1]:,} samples]"
        rules += [rule]

    return rules


def index_of(array, element):
    return np.where(np.array(array) == element)[0][0]


class RuleClassifier:
    def __init__(self, feature_names, rules):
        self.feature_names = feature_names
        self.rules = rules
        self.classes_ = [False, True]

    def predict(self, X):
        df = pd.DataFrame(X, columns=self.feature_names)
        y_pred = np.repeat(False, df.shape[0])
        for rule in self.rules:
            indices = df.query(rule).index
            y_pred[indices] = True
        return y_pred

    def query(self, X):
        df = pd.DataFrame(X, columns=self.feature_names)
        results = []
        for rule in self.rules:
            results.append(df.query(rule))
        return pd.concat(results).drop_duplicates()


class CombinationClassifier:
    def __init__(self, comparer, label):
        self.comparer = comparer
        self.label_explain_a, self.label_explain_b = literal_eval(label)

    def predict(self, X):
        return (self.comparer.clf_a.predict(X) == self.label_explain_a) & \
               (self.comparer.clf_b.predict(X) == self.label_explain_b)
