from typing import List

import diro2c
from comparers import ModelComparer
from data_generation.global_data_generation import global_data_generator
from data_generation.helper import prepare_df
from data_generation.neighborhood_generation import neighbor_generator
from enums.diff_classifier_method_type import diff_classifier_method_type
from sklearn.tree import plot_tree
import rule_extractor
import pandas as pd
import numpy as np
from util import index_of, CombinationClassifier
import matplotlib.pyplot as plt


class PrescaleClassifier:
    def __init__(self, clf, factors):
        self.clf = clf
        self.factors = factors

    def predict(self, X):
        return self.clf.predict(X * self.factors)


class ConstantClassifier:
    def predict(self, X):
        return np.repeat(False, X.shape[0])


class WrappedDiro2C:

    def __init__(self, X, comparer, focus_class=None, scale_features=None):
        """Wrapper to generate diro2c explanations
        scale_features : if specified, scale features with this array of factors prior to generating the explanation
                         and scale back afterwards to tackle the issue with integer-only support of diro2c
        focus_class : if specified, generate a binary diff clf explanation in a one-vs-rest style for this class
        """
        if scale_features is not None:
            self.factors = np.repeat(1.0, X.shape[1])
            for feature, factor in scale_features.items():
                idx = np.where(comparer.feature_names == feature)[0][0]
                self.factors[idx] = factor
            self.X = X * self.factors
        else:
            self.factors = None
            self.X = X

        self.focus_class = focus_class
        if self.focus_class is not None:
            self.clf_a = CombinationClassifier(comparer, self.focus_class)
            self.clf_b = ConstantClassifier()
            self.method = diff_classifier_method_type.binary_diff_classifier
            self.class_names = np.array(['not ' + self.focus_class, self.focus_class])
        else:
            self.clf_a = comparer.clf_a
            self.clf_b = comparer.clf_b
            self.method = diff_classifier_method_type.multiclass_diff_classifier
            self.class_names = comparer.class_names

        if self.factors is not None:
            self.clf_a = PrescaleClassifier(self.clf_a, 1/self.factors)
            self.clf_b = PrescaleClassifier(self.clf_b, 1/self.factors)

        self.feature_names = comparer.feature_names

        d = dict([(feature_name, feature_data)
                  for feature_name, feature_data
                  in zip(self.feature_names, [x for x in self.X.T])])
        d |= {'y': self.clf_a.predict(self.X).astype(str)}
        df = pd.DataFrame(d)
        self.dataset = prepare_df(df, 'test', 'y')

    def generate_local_explanation(self, idx_explain, gn_population_size=5000):
        return self._generate_explanation(neighbor_generator.get_modified_genetic_neighborhood, gn_population_size,
                                          idx_explain)

    def generate_global_explanation(self, gn_population_size=5000):
        return self._generate_explanation(global_data_generator.get_global_mod_genetic_neighborhood_dataset,
                                          gn_population_size)

    def _generate_explanation(self, generation_func, gn_population_size, idx_explain=0):
        explanation = diro2c.recognize_diff(idx_explain, self.X, self.dataset, self.clf_a, self.clf_b, self.method,
                                            generation_func, gn_population_size=gn_population_size)
        if self.factors is not None:
            X = explanation['binary_diff_classifer']['evaluation_info']['X'].astype(float) / self.factors
            explanation['binary_diff_classifer']['evaluation_info']['X'] = X
        explanation['class_names'] = self.class_names
        return explanation


def plot_diro2c_2d(explanation, feature_x, feature_y, xlim=None, ylim=None, highlight=None):
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
    for class_idx, class_label in enumerate(explanation['class_names']):
        mask = y_diff == class_idx
        ax.scatter(X_diff[mask, idx_x], X_diff[mask, idx_y], label=class_label, alpha=0.5)
    if highlight is not None:
        ax.scatter(highlight[idx_x], highlight[idx_y], color='k', marker='x')
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
    plot_tree(dc_full, feature_names=feature_names, class_names=explanation['class_names'], **kwargs)


def _get_decision_tree(explanation):
    dc_full = explanation['binary_diff_classifer']['dc_full']
    return dc_full


def print_diro2c_rules(explanation):
    feature_names = _get_feature_names(explanation)
    dc_full = _get_decision_tree(explanation)
    rule_extractor.print_rules_for_binary(dc_full, feature_names, explanation['class_names'], explanation['class_names'][1])
