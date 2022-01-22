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


def search_max_depth_parameter(explanation, X, y_true, start=2, stop=None):
    feature_names = get_feature_names(explanation)
    indices = np.where(y_true)[0]
    max_depth = start
    parameters = []
    metrics = []
    while stop is None or max_depth <= stop:
        train_surrogate_tree(explanation, max_depth=max_depth)
        model = get_surrogate_tree(explanation)

        constraints, rules, class_occurences, instance_indices_per_rule = extract_rules(explanation, X, y_true)

        if len(rules) > 0:
            rclf = RuleClassifier(feature_names, rules)
            pred_rules = pd.Series(rclf.apply(X[indices]), index=indices)
            pred_rules = pred_rules[pred_rules != 0]
            rule_ids = np.unique(pred_rules) - 1
            n_rules = len(rule_ids)
            n_constraints = np.sum(~np.isnan(constraints[rule_ids]))

            if n_rules > 0:
                rclf = RuleClassifier(feature_names, np.array(rules)[rule_ids])
                y_pred = rclf.predict(X)
                precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred)

                parameters.append(max_depth)
                metrics.append((precision[1], recall[1], n_rules, n_constraints))

        if max_depth > model.get_depth():
            break

        max_depth += 1

    return pd.DataFrame(metrics, index=parameters, columns=['precision', 'recall', 'rules', 'constraints'])


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


def plot_surrogate_tree(explanation, precision=3, figsize=(14, 14), node_ids=False):
    model = get_surrogate_tree(explanation)
    feature_names = get_feature_names(explanation)
    surrogate_tree.plot_surrogate_tree(model, feature_names, precision=precision, figsize=figsize, node_ids=node_ids)


def get_surrogate_tree(explanation):
    return explanation['binary_diff_classifer']['dc_full']


def get_feature_names(explanation):
    return explanation['binary_diff_classifer']['evaluation_info']['df_diff'].columns[1:].to_numpy()


def get_feature_importances(explanation):
    return surrogate_tree.get_feature_importances(get_surrogate_tree(explanation))


def plot_feature_importances(explanation, feature_importances, feature_order, figsize=(5, 2)):
    feature_names = get_feature_names(explanation)
    surrogate_tree.plot_feature_importances(feature_names, feature_importances, feature_order, figsize)


def extract_rules(explanation, X, y, label=1):
    model = get_surrogate_tree(explanation)
    feature_names = get_feature_names(explanation)
    constraints, rules, class_occurences, _, instance_indices_per_rule =\
        surrogate_tree.extract_rules(model, feature_names, [label], X, y)
    return constraints, rules, class_occurences, instance_indices_per_rule


def print_rules(rules, class_occurences):
    for idx, (rule, class_occurences) in enumerate(zip(rules, class_occurences), 1):
        print(f'{idx}. {rule} {class_occurences.astype(int).tolist()}')


def evaluate_generated_data(explanation):
    model = get_surrogate_tree(explanation)
    X, y = get_generated_data(explanation)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))


def evaluate(explanation, X, y):
    model = get_surrogate_tree(explanation)
    surrogate_tree.evaluate(model, X, y)


def plot_tree_leafs_2d(explanation, comparer, X, feature_x=0, feature_y=1, figsize=(7, 7)):
    model = get_surrogate_tree(explanation)
    class_names = [str(label) for label in model.classes_]
    mclass_diff = comparer.predict_mclass_diff(X)
    surrogate_tree.plot_tree_leafs_for_class(model, class_names, class_names[-1], X, mclass_diff, comparer.class_names,
                                             comparer.feature_names, feature_x, feature_y, figsize)
