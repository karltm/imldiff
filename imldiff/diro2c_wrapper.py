import diro2c
from data_generation.helper import prepare_df
from data_generation.neighborhood_generation import neighbor_generator
from enums.diff_classifier_method_type import diff_classifier_method_type
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import plot_tree
import rule_extractor
import pandas as pd
import numpy as np
from util import index_of
import matplotlib.pyplot as plt


class_names = ['no_diff', 'diff']


class CombinationClassifier:
    def __init__(self, comparer, label_explain_a, label_explain_b):
        self.comparer = comparer
        self.label_explain_a = label_explain_a
        self.label_explain_b = label_explain_b

    def predict(self, X):
        return (self.comparer.clf_a.predict(X) == self.label_explain_a) & \
               (self.comparer.clf_b.predict(X) == self.label_explain_b)


class ConstantClassifier:
    def predict(self, X):
        return np.repeat(False, X.shape[0])


def generate_diro2c_explanation(X, idx_explain, comparer, confusion_class):
    label_explain_a, label_explain_b = comparer.class_tuples[index_of(comparer.class_names, confusion_class)]
    clf_a = CombinationClassifier(comparer, label_explain_a, label_explain_b)
    clf_b = ConstantClassifier()

    d = dict([(feature_name, feature_data)
              for feature_name, feature_data
              in zip(comparer.feature_names, [x for x in X.T])])
    d |= {'y': clf_a.predict(X).astype(str)}
    df = pd.DataFrame(d)
    dataset = prepare_df(df, 'test', 'y')

    return diro2c.recognize_diff(idx_explain, X, dataset, clf_a, clf_b,
                                 diff_classifier_method_type.binary_diff_classifier,
                                 data_generation_function=neighbor_generator.get_modified_genetic_neighborhood)


def plot_diro2c_2d(explanation, feature_x, feature_y, xlim=None, ylim=None):
    X_diff, y_diff = _get_X_and_y(explanation)
    feature_names = _get_feature_names(explanation)
    if isinstance(feature_x, str) and isinstance(feature_y, str):
        idx_x, idx_y = index_of(feature_names, feature_x), index_of(feature_names, feature_y)
    else:
        idx_x, idx_y = feature_x, feature_y
        feature_x = feature_names[idx_x]
        feature_y = feature_names[idx_y]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    for class_idx, class_label in [(0, 'no_diff'), (1, 'diff')]:
        mask = y_diff == class_idx
        ax.scatter(X_diff[mask, idx_x], X_diff[mask, idx_y], label=class_label, alpha=0.5)
    ax.legend()


def _get_X_and_y(explanation):
    evaluation_info = explanation['binary_diff_classifer']['evaluation_info']
    return evaluation_info['X'], evaluation_info['y']


def _get_feature_names(explanation):
    evaluation_info = explanation['binary_diff_classifer']['evaluation_info']
    return evaluation_info['df_diff'].columns[1:].to_numpy()


def plot_diro2c_tree(explanation, **kwargs):
    feature_names = _get_feature_names(explanation)
    dc_full = _get_decision_tree(explanation)
    plot_tree(dc_full, feature_names=feature_names, class_names=class_names, **kwargs)


def _get_decision_tree(explanation):
    dc_full = explanation['binary_diff_classifer']['dc_full']
    return dc_full


def print_diro2c_rules(explanation):
    feature_names = _get_feature_names(explanation)
    dc_full = _get_decision_tree(explanation)
    rule_extractor.print_rules_for_binary(dc_full, feature_names, class_names, 'diff')
