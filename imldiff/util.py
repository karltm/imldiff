import re
from ast import literal_eval
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from shap.plots import colors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, precision_recall_fscore_support
from shap.plots.colors import red_blue


settings = {
    'contour_legend_location': 'best',
    'scatter_legend_location': 'best'
}
plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
DEFAULT_PLOT_SIZE = (6, 5)

def get_index_and_name(names, index_or_name):
    if isinstance(index_or_name, str):
        index = list(names).index(index_or_name)
        name = index_or_name
    else:
        index = index_or_name
        name = names[index]
    return index, name


class Counterfactual:
    def __init__(self, feature, value, is_direction_up, outcomes):
        self.feature = feature
        self.value = value
        self.is_direction_up = is_direction_up
        self.outcomes = outcomes

    def is_including_class(self, class_):
        return class_ in self.outcomes.keys()

    def __repr__(self):
        outcomes = '.'.join([f'{count}x {label}' for label, count in self.outcomes.items()])
        return f'{self.value}: {outcomes}'


def find_counterfactuals(comparer, X, X_context, feature_precisions, label):
    counterfactuals = {}
    for feature_idx, feature in enumerate(comparer.feature_names):
        cf_up = _find_counterfactual(comparer, X, X_context, feature_precisions, feature, label, upwards=True)
        cf_down = _find_counterfactual(comparer, X, X_context, feature_precisions, feature, label, upwards=False)
        cfs = []
        if cf_down is not None and not cf_down.is_including_class(label):
            cfs.append(cf_down)
        if cf_up is not None and not cf_up.is_including_class(label):
            cfs.append(cf_up)
        counterfactuals[feature] = cfs
    return counterfactuals


def _find_counterfactual(comparer, X, X_context, feature_precisions, feature, label, upwards):
    feature_idx, feature_name = comparer.check_feature(feature)
    X_mod = X.copy()
    precision = feature_precisions[feature_idx]
    if upwards:
        sign = 1
        start = round(X[:, feature_idx].max(), precision)
        limit = X_context[:, feature_idx].max()
    else:
        sign = -1
        start = round(X[:, feature_idx].min(), precision)
        limit = X_context[:, feature_idx].min()
    step = sign * 10 ** -precision
    for value in np.arange(start, limit + step, step):
        X_mod[:, feature_idx] = round(value, precision)
        y_mod = comparer.predict_mclass_diff(X_mod)
        if np.all(y_mod != label):
            class_counts = dict(pd.Series(comparer.class_names[y_mod]).value_counts())
            return Counterfactual(feature_name, value=X_mod[0, feature_idx], is_direction_up=upwards, outcomes=class_counts)
    return None


def counterfactuals_to_constraint_matrix(feature_names, feature_precisions, include_features, counterfactuals):
    constraint = np.full((len(feature_names), 2), np.nan)
    for feature in include_features:
        feature_idx, feature_name = get_index_and_name(feature_names, feature)
        cfs = counterfactuals.get(feature_name, None)
        for cf in cfs:
            boundary_idx = 1 if cf.is_direction_up else 0
            constraint[feature_idx, boundary_idx] = cf.value

    # because the upper boundary is exclusive
    for row_idx in range(len(constraint)):
        precision = feature_precisions[row_idx]
        value = constraint[row_idx, 1] - 10 ** -precision
        value = round(value, precision)
        constraint[row_idx, 1] = value

    return constraint


def calc_ground_truth_constraints(comparer, feature_precisions, X, label):
    label, class_name = comparer.check_class(label)
    mclass_diff = comparer.predict_mclass_diff(X)
    instance_indices = np.where(mclass_diff == label)[0]
    constraints = []
    for instance in X[instance_indices]:
        counterfactuals = find_counterfactuals(comparer, instance.reshape((1, -1)), X, feature_precisions, label)
        constraint = counterfactuals_to_constraint_matrix(comparer.feature_names, feature_precisions,
                                                          comparer.feature_names, counterfactuals)
        constraints.append(constraint)
    constraints = np.array(constraints)
    return instance_indices, constraints


def evaluate_counterfactual_fidelity(comparer, ground_truth_instance_indices, ground_truth,
                                     instance_indices_per_rule, constraints):
    constraint_tuple = []
    for instance_indices, constraint in zip(instance_indices_per_rule, constraints):
        for instance_idx in instance_indices:
            ground_truth_idx = list(ground_truth_instance_indices).index(instance_idx)
            constraint_tuple.append((ground_truth[ground_truth_idx], constraint, instance_idx))
    return calc_constraint_error(comparer.feature_names, *zip(*constraint_tuple))


def calc_constraint_error(feature_names, constraints_true, constraints_test, indices=None):
    if indices is None:
        indices = range(1, len(constraints_test) + 1)
    errors = []
    n_boundaries_miss = np.zeros(len(feature_names))
    n_boundaries_true = np.zeros(len(feature_names))
    n_boundaries_add = np.zeros(len(feature_names))
    n_boundaries_test = np.zeros(len(feature_names))
    for constraint_true, constraint_test in zip(constraints_true, constraints_test):
        errors.append(constraint_test - constraint_true)
        n_boundaries_miss += np.nansum(~np.isnan(constraint_true) & np.isnan(constraint_test), 1)
        n_boundaries_true += np.nansum(~np.isnan(constraint_true), 1)
        n_boundaries_add += np.nansum(np.isnan(constraint_true) & ~np.isnan(constraint_test), 1)
        n_boundaries_test += np.nansum(~np.isnan(constraint_test), 1)
    errors = np.concatenate(errors, 1)
    errors = pd.DataFrame(errors.T, columns=feature_names,
                          index=pd.MultiIndex.from_product([indices, ['lower', 'upper']]))
    rmse = np.sqrt(np.square(errors).mean())
    boundary_miss_rate = pd.Series(n_boundaries_miss / n_boundaries_true, index=feature_names)
    boundary_add_rate = pd.Series(n_boundaries_add / n_boundaries_test, index=feature_names)
    statistics = pd.DataFrame([rmse, boundary_miss_rate, boundary_add_rate], index=['RMSE', 'BMR', 'BAR']).T
    return errors, statistics


def plot_decision_boundary(X, z=None, title=None, feature_names=None, X_display=None, predict=None,
                           idx_x=0, idx_y=1, class_names=None, zlim=None, mesh_step_size=.5,
                           fig=None, ax=None, xlim=None, ylim=None, predict_value_names=None, predict_value_order=None,
                           show_contour_legend=False, z_label=None, show_colorbar=True, **kwargs):
    """
    - X: instances to plot
    - z: color of instances
    - title: for figure
    - feature_names
    - predict: draw contours for this function (only if number of features is 2)
    - idx_x: index of feature to plot on x axis
    - idx_y: index of feature to plot on y axis
    - class_names: set this to a list of class names if predict returns labels
    - zlim: set this to the range of values if predict returns a continuous variable, e.g. (0, 1)
    - fig, ax
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_PLOT_SIZE)

    if class_names is None and zlim is None:
        if z.dtype == int or z.dtype == bool:
            class_names = np.unique(z)
        else:
            zlim = np.min(z), np.max(z)

    if X_display is None:
        X_display = X

    if z is None:
        z = predict(X)

    draw_contours = predict is not None and X.shape[1] == 2
    if draw_contours:
        if xlim is None:
            xlim = X[:, idx_x].min() - .5, X[:, idx_x].max() + .5
        if ylim is None:
            ylim = X[:, idx_y].min() - .5, X[:, idx_y].max() + .5
        xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], mesh_step_size), np.arange(ylim[0], ylim[1], mesh_step_size))
        z_pred = predict(np.c_[xx.ravel(), yy.ravel()])
        z_pred = z_pred.reshape(xx.shape)

    if class_names is not None:
        predict_value_names = class_names if predict_value_names is None else predict_value_names
        predict_value_order = range(len(predict_value_names)) if predict_value_order is None else predict_value_order
        legend1 = None
        if draw_contours:
            levels = np.arange(len(predict_value_names) + 1)
            cs = ax.contourf(xx, yy, z_pred + 0.5, levels, colors=plt_colors, alpha=.8)
            proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
                     for pc in cs.collections]
            if show_contour_legend:
                proxy = [proxy[idx] for idx in predict_value_order]
                predict_value_names = [predict_value_names[idx] for idx in predict_value_order]
                legend1 = ax.legend(proxy, predict_value_names, title=z_label, loc=settings['contour_legend_location'])
        for class_idx, class_ in enumerate(class_names):
            X_ = X_display[z == class_idx, :]
            if X_.shape[0] == 0:
                continue
            ax.scatter(X_[:, idx_x], X_[:, idx_y], color=plt_colors[class_idx], edgecolors='k', label=str(class_), **kwargs)
        ax.legend(loc=settings['scatter_legend_location'])
        if legend1 is not None:
            plt.gca().add_artist(legend1)
    else:
        if draw_contours:
            levels = np.linspace(zlim[0], zlim[1], 21)
            cs = ax.contourf(xx, yy, z_pred, levels, cmap=colors.red_blue, alpha=.8)
            if show_colorbar:
                fig.colorbar(cs, ax=ax, shrink=0.9, label=z_label)
        ax.scatter(X[:, idx_x], X[:, idx_y], c=z, cmap=colors.red_blue, vmin=zlim[0], vmax=zlim[1], edgecolors='k', **kwargs)

    if feature_names is not None:
        ax.set_xlabel(feature_names[idx_x])
        ax.set_ylabel(feature_names[idx_y])
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def draw_colorbar_if_necessary(zlim, fig=None, axs=None):
    norm = plt.Normalize(*zlim)
    sm = plt.cm.ScalarMappable(cmap=red_blue, norm=norm)
    sm.set_array([])
    fig = plt.gca().figure if fig is None else fig
    axs = fig.axes if axs is None else axs
    fig.colorbar(sm, ax=axs, shrink=0.9, pad=0.01)


class CombinationClassifier:
    def __init__(self, comparer, label):
        self.comparer = comparer
        self.label_explain_a, self.label_explain_b = literal_eval(label)

    def predict(self, X):
        return (self.comparer.clf_a.predict(X) == self.label_explain_a) & \
               (self.comparer.clf_b.predict(X) == self.label_explain_b)


def constraint_matrix_to_rules(constraints, feature_names, feature_order=None, precisions=None, latex=False):
    if feature_order is None:
        feature_order = range(len(feature_names))
    rules = []
    lt = '<'
    leq = '\leq' if latex else '<='
    gt = '>'
    land = '\land' if latex else 'and'
    for constraint in constraints:
        terms = []
        for feature_idx in feature_order:
            feature_name = escape_feature_name(feature_names[feature_idx], latex)
            lower_boundary = constraint[feature_idx, 0]
            upper_boundary = constraint[feature_idx, 1]
            precision = _get_precision(feature_idx, precisions)
            if not np.isnan(lower_boundary):
                if not np.isnan(upper_boundary):
                    terms.append(f'{_fmt(lower_boundary, precision)} {lt} {feature_name} {leq} {_fmt(upper_boundary, precision)}')
                else:
                    terms.append(f'{feature_name} {gt} {_fmt(lower_boundary, precision)}')
            elif not np.isnan(upper_boundary):
                terms.append(f'{feature_name} {leq} {_fmt(upper_boundary, precision)}')
        rules.append(f' {land} '.join(terms))
    return rules


def _get_precision(feature_idx, precisions):
    if precisions is not None:
        return precisions[feature_idx]
    return None


def _fmt(x, precision):
    if precision is None:
        return str(x)
    return ('{:.' + str(precision) + 'f}').format(x)


def escape_feature_name(feature_name, latex=False):
    if latex:
        if feature_name == 'x1':
            feature_name = 'x_1'
        elif feature_name == 'x2':
            feature_name = 'x_2'
        else:
            feature_name = '\\text{' + feature_name + '}'
    else:
        if ' ' in feature_name or '-' in feature_name:
            feature_name = f'`{feature_name}`'
    return feature_name


def get_complexity(constraints):
    n_rules = len(constraints)
    n_constraints = np.sum(~np.isnan(constraints))
    return pd.Series([n_rules, n_constraints], index=['Rules', 'Constraints'])


class RuleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_names, rules):
        self.feature_names = feature_names
        self.rules = rules
        self.classes_ = [False, True]

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)
        else:
            X = pd.DataFrame(X, columns=self.feature_names)
        y_pred = np.repeat(False, X.shape[0])
        rule = ' or '.join(self.rules)
        if rule:
            indices = X.query(rule).index
            y_pred[indices] = True
        return y_pred

    def apply(self, X):
        """For each instance, return the number of the first rule that applies or 0."""
        df = pd.DataFrame(X, columns=self.feature_names)
        y_pred = np.zeros(df.shape[0])
        for rule_no, rule in enumerate(self.rules, 1):
            indices = df.query(rule).index
            y_pred[indices] = rule_no
        return y_pred.astype(int)


def evaluate(model, X, y, class_names=None):
    if class_names is None:
        class_names = [str(label) for label in model.classes_]
    else:
        class_names = np.array(class_names)[model.classes_]
    y_pred = model.predict(X)
    classes = model.classes_
    df = evaluate_predictions(y, y_pred, classes, class_names)
    return df


def evaluate_predictions(y, y_pred, classes, class_names):
    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(y, y_pred, labels=classes, warn_for=[])
    df = pd.DataFrame(np.array((precisions, recalls, f1_scores, supports)).T,
                      columns=['Precision', 'Recall', 'F1 Score', 'Support'],
                      index=class_names)
    df['Support'] = df['Support'].astype(int)
    return df
