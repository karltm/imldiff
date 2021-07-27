from typing import Iterable
import shap
from shap.maskers import Independent
from shap.utils import hclust_ordering
from shap.plots import colors
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from types import SimpleNamespace
from comparers import ModelComparer
from IPython.display import display


def generate_shap_explanations(comparer: ModelComparer, X: np.ndarray, X_display: np.ndarray = None,
                               explanation_types: Iterable[str] = None, space_types: Iterable[str] = None,
                               algorithm = 'auto', masker: shap.maskers.Masker = None):
    """ Generate SHAP values for difference classifier

    :param comparer: model comparison helper
    :param X: dataset to generate SHAP values for
    :param X_display: dataset with same shape as X, which is used for plots
                      and may contain descriptive categorical values
    :param explanation_types: list of types of explanations to generate
                              'indiv' - individual models' SHAP values
                              'indiv_diff' - difference between individual models' SHAP values
                              'bin_diff' - SHAP values of binary difference classifier
                              'mclass_diff' - SHAP values of multiclass difference classifier
    :param space_types: list of types of spaces for which to generate SHAP values for
                        'labels' - predicted labels (hard decision boundary)
                        'proba' - predicted probabilities (soft decision boundary)
                        'log_odds' - predicted log odds (soft decision boundary)
    :param algorithm: SHAP value generation algorithm. See shap.Explainer for possible values
    :param masker: If you want to customize the masker used during SHAP value generation that masks out features.
                   Default: shap.maskers.Independent(data=X)

    :return tuple consisting of:
            explanations - sliceable container of SHAP values for all requested kinds
            indices_nonfinite_predictions - indices of predictions that were filtered because
                                            of non-finite model predictions
            explanations_nonfinite - SHAP values that were filtered because of non-finite SHAP values

            the sliceable container contains in the first level the explanation types
            and in the second level the space types and for individual explanations another level for each model.
            To access the probability SHAP values of classifier A: explanations.indiv.proba.A
    """
    if space_types is None:
        space_types = ['labels', 'proba', 'log_odds']
    if explanation_types is None:
        explanation_types = ['indiv', 'indiv_diff', 'bin_diff', 'mclass_diff']
    if 'indiv_diff' in explanation_types and not 'indiv' in explanation_types:
        raise Exception('Cannot calculate shap value differences without the individual models\' shap values')

    instance_names = np.arange(X.shape[0])
    explainers = _make_shap_explainers(explanation_types, space_types, comparer, X, algorithm, masker)
    instance_names, X, X_display, indices_nonfinite_predictions = \
        _filter_instances_with_nonfinite_predictions(explainers, instance_names, X, X_display)
    explanations = VariousTypeExplanationsNamespace()
    _make_shap_values(explanations, X, X_display, explainers)
    _derive_shap_values(explanations, comparer, explanation_types)
    explanations, explanations_nonfinite = _filter_nonfinite_shap_values(explanations, instance_names)
    return explanations, indices_nonfinite_predictions, explanations_nonfinite


def _make_shap_explainers(explanation_types, space_types, comparer, X, algorithm, masker=None):
    if masker is None:
        masker = Independent(data=X)
    feature_names = comparer.feature_names

    explainers = SimpleNamespace()
    if 'indiv' in explanation_types:
        ns = SimpleNamespace()
        if 'labels' in space_types:
            subns = SimpleNamespace()
            for classifier_name, predict in comparer.predict_one_hot_functions.items():
                explainer = _make_shap_explainer(predict, masker=masker, algorithm=algorithm,
                                                 feature_names=feature_names, output_names=comparer.base_class_names)
                subns.__dict__[classifier_name] = explainer
            ns.labels = subns
        if 'proba' in space_types:
            subns = SimpleNamespace()
            for classifier_name, predict in comparer.predict_proba_functions.items():
                explainer = _make_shap_explainer(predict, masker=masker, algorithm=algorithm,
                                                 feature_names=feature_names, output_names=comparer.base_class_names)
                subns.__dict__[classifier_name] = explainer
            ns.proba = subns
        if 'log_odds' in space_types:
            subns = SimpleNamespace()
            for classifier_name, predict in comparer.predict_log_odds_functions.items():
                explainer = _make_shap_explainer(predict, masker=masker, algorithm=algorithm,
                                                 feature_names=feature_names, output_names=comparer.base_class_names)
                subns.__dict__[classifier_name] = explainer
            ns.log_odds = subns
        explainers.indiv = ns
    if 'bin_diff' in explanation_types:
        ns = SimpleNamespace()
        if 'labels' in space_types:
            ns.labels = _make_shap_explainer(comparer.predict_bin_diff, masker=masker, output_names='diff',
                                             algorithm=algorithm, feature_names=feature_names)
        if 'proba' in space_types:
            ns.proba = _make_shap_explainer(comparer.predict_bin_diff_proba, masker=masker, output_names='diff',
                                            algorithm=algorithm, feature_names=feature_names)
        if 'log_odds' in space_types:
            ns.log_odds = _make_shap_explainer(comparer.predict_bin_diff_log_odds, masker=masker, output_names='diff',
                                               algorithm=algorithm, feature_names=feature_names)
        explainers.bin_diff = ns
    if 'mclass_diff' in explanation_types:
        ns = SimpleNamespace()
        if 'labels' in space_types:
            ns.labels = _make_shap_explainer(comparer.predict_mclass_diff_one_hot, masker=masker,
                                             algorithm=algorithm, feature_names=feature_names,
                                             output_names=comparer.class_names)
        if 'proba' in space_types:
            ns.proba = _make_shap_explainer(comparer.predict_mclass_diff_proba, masker=masker,
                                            algorithm=algorithm, feature_names=feature_names,
                                            output_names=comparer.class_names)
        if 'log_odds' in space_types:
            ns.log_odds = _make_shap_explainer(comparer.predict_mclass_diff_log_odds, masker=masker,
                                               algorithm=algorithm, feature_names=feature_names,
                                               output_names=comparer.class_names)
        explainers.mclass_diff = ns
    return explainers


def _make_shap_explainer(predict, masker, algorithm, feature_names, output_names=None):
    explainer = shap.Explainer(predict, masker=masker, algorithm=algorithm, feature_names=feature_names,
                               output_names=output_names)

    # workaround: not all shap explainers support output_names. so we remember them for later to recover them
    if output_names is not None and explainer.output_names is None:
        explainer.output_names_backup = output_names
    return explainer


def _filter_instances_with_nonfinite_predictions(explainers, instance_names, X, X_display=None):
    filter_mask = np.repeat(True, X.shape[0])
    for explanation_type, explainers_ns in explainers.__dict__.items():
        for space_type, space_sub in explainers_ns.__dict__.items():
            if isinstance(space_sub, shap.Explainer):
                filter_mask &= _make_nonfinite_predictions_mask_for_explainer(space_sub, X)
            elif isinstance(space_sub, SimpleNamespace):
                for sub_type, explainer in space_sub.__dict__.items():
                    filter_mask &= _make_nonfinite_predictions_mask_for_explainer(explainer, X)

    if np.sum(~filter_mask) > 0:
        warnings.warn(f'filtering instances with nonfinite predictions: {instance_names[~filter_mask]}')
        instance_names = instance_names[filter_mask]
        indices_nonfinite_predictions = instance_names[~filter_mask]
        X = X[filter_mask, :]
        if X_display is not None:
            X_display = X_display[filter_mask, :]
    else:
        indices_nonfinite_predictions = np.array([])
    return instance_names, X, X_display, indices_nonfinite_predictions


def _make_nonfinite_predictions_mask_for_explainer(explainer, X):
    y_pred = explainer.model(X)
    if len(y_pred.shape) == 1:
        mask = np.isfinite(y_pred)
    else:
        mask = np.all(np.isfinite(y_pred), axis=1)
    return mask


class BaseExplanationsNamespace(SimpleNamespace):
    """ A sliceable, pickleable container for groups of SHAP values """

    def __getitem__(self, *args, **kwargs):
        return self.__class__(**dict([(k, v.__getitem__(*args, **kwargs)) for k, v in self.__dict__.items()]))

    @property
    def data(self):
        return next(iter(self.__dict__.values())).data

    @property
    def display_data(self):
        return next(iter(self.__dict__.values())).display_data

    @property
    def feature_names(self):
        return next(iter(self.__dict__.values())).feature_names

    @property
    def merged(self):
        return merge_explanations(**self.__dict__)

    def cohorts(self, cohorts):
        new_explanations = self.__class__()
        for k, v in self.__dict__.items():
            new_explanations.__dict__[k] = v.cohorts(cohorts)
        return new_explanations


def merge_explanations(**explanations):
    names = []
    values = []
    base_values = []
    for k, v in explanations.items():
        if isinstance(v, BaseExplanationsNamespace):
            v = v.merged
        v = ensure_shap_values_are_3d(v)
        names += [f'{k}.{name}' for name in v.output_names]
        values.append(v.values)
        base_values.append(v.base_values)
    values = np.concatenate(values, axis=2)
    base_values = np.concatenate(base_values, axis=1)
    first = next(iter(explanations.values()))
    return shap.Explanation(values, base_values, first.data, first.display_data,
                            feature_names=first.feature_names, output_names=names)


def ensure_are_shap_values(shap_values):
    if isinstance(shap_values, BaseExplanationsNamespace):
        return shap_values.merged
    return shap_values


def ensure_shap_values_are_3d(shap_values):
    shap_values = ensure_are_shap_values(shap_values)
    if len(shap_values.shape) == 3:
        return shap_values
    if len(shap_values.shape) == 2:
        if shap_values.output_names is not None:
            names = [shap_values.output_names]
        else:
            names = None
        values = shap_values.values.reshape((shap_values.shape[0], shap_values.shape[1], 1))
        base_values = shap_values.base_values.reshape((shap_values.shape[0], 1))
        return shap.Explanation(values, base_values, shap_values.data, shap_values.display_data,
                                feature_names=shap_values.feature_names, output_names=names)
    raise Exception(f'invalid dimensions: {len(shap_values.shape)}')


def ensure_all_shap_values_are_3d(*shap_values):
    return tuple([ensure_shap_values_are_3d(s) for s in shap_values])


class SameTypeExplanationsNamespace(BaseExplanationsNamespace):

    @property
    def shape(self):
        return next(iter(self.__dict__.values())).shape

    @property
    def output_names(self):
        return next(iter(self.__dict__.values())).output_names


class VariousTypeExplanationsNamespace(BaseExplanationsNamespace):

    @property
    def shape(self):
        return next(iter(self.__dict__.values())).shape[:2]


def _make_shap_values(explanations, X, X_display, explainers):
    for explanation_type, explainers_ns in explainers.__dict__.items():
        ns = SameTypeExplanationsNamespace()
        for space_type, space_sub in explainers_ns.__dict__.items():
            if isinstance(space_sub, shap.Explainer):
                ns.__dict__[space_type] = _make_shap_values_for_explainer(X, X_display, space_sub)
            elif isinstance(space_sub, SimpleNamespace):
                subns = SameTypeExplanationsNamespace()
                for sub_type, explainer in space_sub.__dict__.items():
                    subns.__dict__[sub_type] = _make_shap_values_for_explainer(X, X_display, explainer)
                ns.__dict__[space_type] = subns
        explanations.__dict__[explanation_type] = ns


def _make_shap_values_for_explainer(X, X_display, explainer):
    shap_values = explainer(X)
    if X_display is not None:
        shap_values.display_data = X_display
    # workaround: restore output_names for explainers that don't support them
    if hasattr(explainer, 'output_names_backup'):
        shap_values = shap.Explanation(shap_values.values, shap_values.base_values, shap_values.data,
                                       shap_values.display_data, shap_values.instance_names,
                                       shap_values.feature_names, explainer.output_names_backup)
    return shap_values


def _derive_shap_values(explanations, comparer, explanation_types):
    if 'indiv_diff' in explanation_types:
        ns = SameTypeExplanationsNamespace()
        for space_type, space_sub in explanations.indiv.__dict__.items():
            shap_values_a = space_sub.A
            shap_values_b = space_sub.B
            values = shap_values_a.values - shap_values_b.values
            base_values = shap_values_a.base_values - shap_values_b.base_values
            shap_values = shap.Explanation(values, base_values, explanations.data, explanations.display_data,
                                           feature_names=explanations.feature_names,
                                           output_names=comparer.base_class_names)
            ns.__dict__[space_type] = shap_values
        explanations.indiv_diff = ns


def _filter_nonfinite_shap_values(explanations, instance_names):
    filter_mask = np.repeat(True, explanations.shape[0])
    for explanation_type, ns in explanations.__dict__.items():
        for space_type, space_sub in ns.__dict__.items():
            if isinstance(space_sub, shap.Explanation):
                filter_mask &= _make_nonfinite_shap_values_mask(space_sub)
            elif isinstance(space_sub, SimpleNamespace):
                for sub_type, shap_values in space_sub.__dict__.items():
                    filter_mask &= _make_nonfinite_shap_values_mask(shap_values)
    if np.sum(~filter_mask) > 0:
        warnings.warn(f'filtering instances with nonfinite shap values: {instance_names[~filter_mask]}')
        explanations = explanations[filter_mask]
        explanations_nonfinite = explanations[~filter_mask]
    else:
        explanations_nonfinite = explanations[np.repeat(False, explanations.shape[0])]
    return explanations, explanations_nonfinite


def _make_nonfinite_shap_values_mask(shap_values):
    if len(shap_values.shape) == 2:
        return np.all(np.isfinite(shap_values.values), axis=1)
    else:
        return np.all(np.all(np.isfinite(shap_values.values), axis=2), axis=1)


def calc_feature_order(shap_values):
    shap_values = ensure_shap_values_are_3d(shap_values)
    values = np.abs(shap_values.values).mean(axis=2).mean(axis=0)
    feature_order = np.flip(values.argsort())
    feature_importance = shap.Explanation(values, feature_names=shap_values.feature_names)
    return feature_order, feature_importance


def calc_class_order(shap_values):
    if not len(shap_values.shape) == 3:
        raise Exception('only multiclass kinds allowed')
    class_importances = np.abs(shap_values.values).mean(axis=1).mean(axis=0)
    class_order = np.flip(np.argsort(class_importances))
    return class_order, class_importances


def plot_class_importances(class_importances, class_order, class_names):
    df = pd.DataFrame(class_importances[class_order], index=np.array(class_names)[class_order])
    df.plot.bar(title='Class importances', ylabel='mean(|SHAP value|)')


def calc_instance_order(shap_values):
    shap_values = ensure_shap_values_are_3d(shap_values)
    values = shap_values.values.reshape(
        (shap_values.values.shape[0],
         shap_values.values.shape[1] * shap_values.values.shape[2]))
    instance_order = np.argsort(hclust_ordering(values))
    return instance_order


def plot_2d(*shap_values, title=None, x=0, y=1, **kwargs):
    shap_values = ensure_all_shap_values_are_3d(*shap_values)
    ncols = sum([s.shape[2] for s in shap_values])
    nrows = shap_values[0].shape[1]
    fig, axs = plt.subplots(nrows, ncols, figsize=(9*ncols, 9*nrows), constrained_layout=True, sharex=True, sharey=True)
    plot_idx = 0
    fig.suptitle(title, fontsize=16)
    for feature_idx in range(nrows):
        vmax = np.max([np.abs(s[:, feature_idx, :].values).flatten().max(0) for s in shap_values])
        for s in shap_values:
            display_shap_values = s[:, [x, y], :]
            X_display = _get_display_data(display_shap_values)
            for class_idx in range(s.shape[2]):
                ax = axs.flat[plot_idx]
                cs = ax.scatter(X_display[:, 0],
                                X_display[:, 1],
                                c=s.values[:, feature_idx, class_idx],
                                vmin=-vmax, vmax=vmax,
                                cmap=colors.red_blue,
                                **kwargs)
                ax.set_title(f'SHAP-values of {s.feature_names[feature_idx]} '
                             f'for {s.output_names[class_idx]}')
                ax.set_xlabel(display_shap_values.feature_names[0])
                ax.set_ylabel(display_shap_values.feature_names[1])
                plot_idx += 1
        fig.colorbar(cs, ax=ax, shrink=0.9)
    plt.show()


def _get_display_data(shap_values):
    if shap_values.display_data is not None:
        return shap_values.display_data
    else:
        return shap_values.data


def plot_feature_importance_bar(shap_values, title=None, feature_order=None):
    shap_values = ensure_are_shap_values(shap_values)
    if len(shap_values.shape) <= 2:
        return _plot_feature_importance_bar_singleclass(shap_values, title, feature_order)
    elif len(shap_values.shape) == 3:
        return _plot_feature_importance_bar_multiclass(shap_values, title)
    raise Exception(f'invalid dimensions: {shap_values.shape}')


def _plot_feature_importance_bar_singleclass(shap_values, title=None, feature_order=None):
    if feature_order is None:
        if len(shap_values.shape) == 2:
            feature_order = range(shap_values.shape[1])
        elif len(shap_values.shape) == 1:
            feature_order = np.flip(np.argsort(shap_values.values))
    plt.title(title)
    shap.plots.bar(shap_values, order=feature_order, max_display=len(feature_order))


def _plot_feature_importance_bar_multiclass(shap_values, title=None):
    shap_values_list = [values.T for values in shap_values.values.T]
    shap.summary_plot(shap_values_list, shap_values.data,
                      feature_names=shap_values.feature_names,
                      class_names=shap_values.output_names, show=False)
    plt.legend(loc='right')
    plt.title(title)
    plt.show()


def plot_feature_importance_scatter(shap_values, title=None, feature_order=None, class_order=None, **kwargs):
    shap_values = ensure_are_shap_values(shap_values)
    if len(shap_values.shape) == 2:
        return _plot_feature_importance_scatter_singleclass(shap_values, title, feature_order, **kwargs)
    elif len(shap_values.shape) == 3:
        return _plot_feature_importance_scatter_multiclass(shap_values, title, feature_order, class_order, **kwargs)
    raise Exception(f'invalid dimensions: {shap_values.shape}')


def _plot_feature_importance_scatter_singleclass(shap_values, title=None, feature_order=None, **kwargs):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    plt.title(title)
    shap.plots.beeswarm(shap_values, order=feature_order, plot_size=(14, 7), **kwargs)


def _plot_feature_importance_scatter_multiclass(shap_values, title=None, feature_order=None, class_order=None, **kwargs):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    if class_order is None:
        class_order = np.arange(shap_values.shape[2])
    plt.suptitle(title, fontsize='x-large')
    for feature_idx in feature_order:
        new_values = shap_values.values[:, feature_idx, :]
        new_data = np.reshape(np.repeat(shap_values.data[:, feature_idx], shap_values.shape[2]),
                              (shap_values.data.shape[0], shap_values.shape[2]))
        if shap_values.display_data is not None:
            new_display_data = np.reshape(np.repeat(shap_values.display_data[:, feature_idx], shap_values.shape[2]),
                                          (shap_values.data.shape[0], shap_values.shape[2]))
        else:
            new_display_data = None
        new_base_values = shap_values.base_values
        shap_values_ = shap.Explanation(new_values, new_base_values, new_data, new_display_data,
                                        feature_names=shap_values.output_names)
        shap.plots.beeswarm(shap_values_, order=class_order, plot_size=(14, 7), show=False, **kwargs)
        plt.title(shap_values.feature_names[feature_idx])
        plt.show()


def plot_feature_effects(*shap_values, title=None, highlight=None, **kwargs):
    """ Plot marginal effect of each feature vs. its SHAP values per class.

    Further keyword arguments are passed to shap.plots.scatter,
    and may include e.g. color=is_pred_diff, alpha=0.2
    """
    shap_values = ensure_all_shap_values_are_3d(*shap_values)
    ncols = sum([s.shape[2] for s in shap_values])
    nrows = shap_values[0].shape[1]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex='row', sharey='row', figsize=(9 * ncols, 7 * nrows))
    fig.suptitle(title, fontsize='x-large', y=0.91)
    plot_idx = 0
    for feature_idx in range(nrows):
        xmin = np.min([s.data[:, feature_idx].min(0) for s in shap_values])
        xmax = np.max([s.data[:, feature_idx].max(0) for s in shap_values])
        ymin = np.min([s.values[:, feature_idx, :].flatten().min(0) for s in shap_values])
        ymax = np.max([s.values[:, feature_idx, :].flatten().max(0) for s in shap_values])
        for s in shap_values:
            for class_idx in range(s.shape[2]):
                ax = axs if ncols == 1 and nrows == 1 else axs.flat[plot_idx]
                shap.plots.scatter(s[:, feature_idx, class_idx], title=s.output_names[class_idx],
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                   ax=ax, show=False, **kwargs)
                if highlight is not None and np.sum(highlight) > 0:
                    shap.plots.scatter(s[highlight, feature_idx, class_idx], title=s.output_names[class_idx],
                                       xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                       ax=ax, show=False, color='r', hist=False, **kwargs)
                plot_idx += 1
    plt.show()


def plot_forces(shap_values, title=None, instance_order=None, class_order=None, **kwargs):
    """ Create force plots of all instances

    Further keyword arguments are passed to shap plot function
    e.g. link='logit'
    """
    shap_values = ensure_are_shap_values(shap_values)
    if len(shap_values.shape) <= 2:
        return plot_forces_singleclass(shap_values, title, instance_order, **kwargs)
    if len(shap_values.shape) == 3:
        return plot_forces_multiclass(shap_values, instance_order, class_order, **kwargs)
    raise Exception(f'invalid dimensions: {shap_values.shape}')


def plot_forces_singleclass(shap_values, title=None, instance_order=None, **kwargs):
    if instance_order is not None and isinstance(instance_order, np.ndarray):
        instance_order = instance_order.tolist()
    X_display = _get_display_data(shap_values)
    plot = shap.plots.force(
        base_value=shap_values.base_values[0],
        shap_values=shap_values.values,
        features=X_display,
        feature_names=shap_values.feature_names,
        out_names=title,
        ordering_keys=instance_order,
        **kwargs)
    display(plot)


def plot_forces_multiclass(shap_values, instance_order=None, class_order=None, **kwargs):
    if class_order is None:
        class_order = range(shap_values.shape[2])
    if instance_order is not None and isinstance(instance_order, np.ndarray):
        instance_order = instance_order.tolist()
    X_display = _get_display_data(shap_values)
    for class_idx in class_order:
        shap_values_ = shap_values[:, :, class_idx]
        plot = shap.plots.force(
            base_value=shap_values_.base_values[0],
            shap_values=shap_values_.values,
            features=X_display,
            feature_names=shap_values.feature_names,
            out_names=str(shap_values_.output_names),
            ordering_keys=instance_order,
            **kwargs)
        display(plot)


def plot_decision(shap_values, classes=None, **kwargs):
    shap_values = ensure_are_shap_values(shap_values)
    if len(shap_values.shape) == 2:
        plot_decision_singleclass(shap_values, **kwargs)
    elif len(shap_values.shape) == 3:
        if classes is None:
            classes = shap_values.output_names
        for class_ in classes:
            plot_decision_singleclass(shap_values[:, :, class_], **kwargs)
    else:
        raise Exception(f'invalid dimensions: {shap_values.shape}')


def plot_decision_singleclass(shap_values, **kwargs):
    plt.title(shap_values.output_names)
    shap.decision_plot(shap_values.base_values[0], shap_values.values, shap_values.feature_names, **kwargs)


def perform_hierarchical_clustering(shap_values):
    shap_values = ensure_shap_values_are_3d(shap_values)
    values = shap_values.values.reshape(
        (shap_values.values.shape[0],
         shap_values.values.shape[1] * shap_values.values.shape[2]))
    D = sp.spatial.distance.pdist(values, metric='sqeuclidean')
    linkage_matrix = sp.cluster.hierarchy.complete(D)
    return linkage_matrix


def plot_dendrogram(linkage_matrix):
    fig, ax = plt.subplots(figsize=(7, 7))
    sp.cluster.hierarchy.dendrogram(linkage_matrix, orientation='right', ax=ax, no_labels=True)
    ax.set_title('Dendrogram')
    plt.show()


def extract_clustering(linkage_matrix, n_clusters):
    cluster_names = np.array([f'c{idx}' for idx in range(1, n_clusters+1)])
    clustering = sp.cluster.hierarchy.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    clustering -= 1
    return clustering, cluster_names


def get_class_occurences_in_clusters(explanations_clustered, cluster_names, comparer):
    occurences = pd.DataFrame(np.zeros((len(cluster_names), comparer.classes.shape[0]), dtype=int),
                              index=cluster_names, columns=comparer.class_names)
    for cluster, data in explanations_clustered.mclass_diff.labels.data.cohorts.items():
        mclass_diff_ = comparer.predict_mclass_diff(data)
        indices, counts = np.unique(mclass_diff_, return_counts=True)
        occurences.loc[cluster, :].iloc[indices] = counts
    has_diff_classes = occurences.loc[:, comparer.difference_class_names].sum(1) > 0
    clusters_of_interest = occurences.index[has_diff_classes].to_numpy()
    return occurences, clusters_of_interest


def plot_feature_influence_comparison(shap_values, instances_mask, feature_order=None, class_order=None):
    shap_values = ensure_shap_values_are_3d(shap_values)
    shap_values = shap_values[:, feature_order][:, :, class_order]
    confused_values = shap_values[instances_mask, :, :].mean(0).values
    if np.sum(~instances_mask) > 0:
        not_confused_values = shap_values[~instances_mask, :, :].mean(0).values
    else:
        not_confused_values = np.zeros(confused_values.shape)
    for feature_name, cv, ncv in zip(shap_values.feature_names, confused_values, not_confused_values):
        df = pd.DataFrame([cv, ncv], columns=shap_values.output_names, index=['confused', 'not confused'])
        df.plot.bar(title=feature_name, ylabel='mean(SHAP value)')
