import diro2c
import surrogate_tree
from data_generation.global_data_generation import global_data_generator
from data_generation.helper import prepare_df
from data_generation.neighborhood_generation import neighbor_generator
from enums.diff_classifier_method_type import diff_classifier_method_type
import pandas as pd
import numpy as np
from util import CombinationClassifier, RuleClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support


class ConstantClassifier:
    def predict(self, X):
        return np.repeat(False, X.shape[0])


class WrappedDiro2C:

    def __init__(self, X, comparer, focus_class=None):
        """Wrapper to generate diro2c explanations
        focus_class : if specified, generate a binary diff clf explanation in a one-vs-rest style for this class
        """
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
        return diro2c.recognize_diff(idx_explain, self.X, self.dataset, self.clf_a, self.clf_b, self.method,
                                     generation_func, gn_population_size=gn_population_size)


def plot_diro2c_2d(explanation, feature_x, feature_y, xlim=None, ylim=None, highlight=None):
    model = get_surrogate_tree(explanation)
    X, y = get_generated_data(explanation)
    feature_names = get_feature_names(explanation)
    if isinstance(feature_x, str) and isinstance(feature_y, str):
        idx_x, idx_y = list(feature_names).index(feature_x), list(feature_names).index(feature_y)
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
    for label in model.classes_:
        mask = y == label
        ax.scatter(X[mask, idx_x], X[mask, idx_y], label=str(label), alpha=0.5)
    if highlight is not None:
        ax.scatter(highlight[idx_x], highlight[idx_y], color='k', marker='x')
    ax.legend()


def train_surrogate_tree(explanation, max_depth=None):
    X, y = get_generated_data(explanation)
    model = surrogate_tree.train_surrogate_tree(X, y, max_depth)
    set_surrogate_tree(explanation, model)


def get_generated_data(explanation):
    evaluation_info = explanation['binary_diff_classifer']['evaluation_info']
    X, y = evaluation_info['X'], evaluation_info['y']
    return X, y


def set_generated_data(explanation, X, y):
    evaluation_info = explanation['binary_diff_classifer']['evaluation_info']
    evaluation_info['X'] = X
    evaluation_info['y'] = y


def set_surrogate_tree(explanation, model):
    explanation['binary_diff_classifer']['dc_full'] = model


def get_surrogate_tree(explanation):
    return explanation['binary_diff_classifer']['dc_full']


def get_feature_names(explanation):
    return explanation['binary_diff_classifer']['evaluation_info']['df_diff'].columns[1:].to_numpy()


def evaluate_generated_data(explanation):
    model = get_surrogate_tree(explanation)
    X, y = get_generated_data(explanation)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))


def evaluate(explanation, X, y):
    model = get_surrogate_tree(explanation)
    return surrogate_tree.evaluate(model, X, y)


def get_pruned_trees(explanation):
    tree = get_surrogate_tree(explanation)
    X_train, y_train = get_generated_data(explanation)
    return surrogate_tree.get_pruned_trees(tree, X_train, y_train)


def eval_diro2c(explanation_per_class, X_test, y_test, class_names):
    metrics = []
    for class_name, explanation in explanation_per_class.items():
        y_true = class_names[y_test] == class_name
        trees = get_pruned_trees(explanation)
        results = surrogate_tree.eval_trees(trees, get_feature_names(explanation), trees[0].classes_, X_test=X_test, y_test=y_true)
        class_metrics = results.loc[1, :].copy()
        class_metrics['Label'] = class_name
        metrics.append(class_metrics)
    return pd.concat(metrics)
