from sklearn.decomposition import PCA
import shap
from shap.maskers import Independent
import numpy as np
import warnings
from types import SimpleNamespace


class SliceableNamespace(SimpleNamespace):

    def __getitem__(self, *args, **kwargs):
        return self.__class__(**dict([(k, v.__getitem__(*args, **kwargs)) for k, v in self.__dict__.items()]))


class SameTypeExplanationsNamespace(SliceableNamespace):

    @property
    def shape(self):   
        return next(iter(self.__dict__.values())).shape

    @property
    def data(self):
        return next(iter(self.__dict__.values())).data

    @property
    def display_data(self):
        return next(iter(self.__dict__.values())).display_data


class VariousTypeExplanationsNamespace(SliceableNamespace):

    @property
    def shape(self):   
        return next(iter(self.__dict__.values())).shape[:2]

    @property
    def data(self):
        return next(iter(self.__dict__.values())).data

    @property
    def display_data(self):
        return next(iter(self.__dict__.values())).display_data


def generate_shap_explanations(comparer, X, X_display=None, explanation_types=None, space_types=None,
                               algorithm='auto', masker=None):
    if space_types is None:
        space_types = ['labels', 'proba', 'log_odds']
    if explanation_types is None:
        explanation_types = ['indiv', 'indiv_diff', 'bin_diff', 'mclass_diff']
    if 'indiv_diff' in explanation_types and not 'indiv' in explanation_types:
        raise Exception('Cannot calculate shap value differences without the individual models\' shap values')

    instance_names = np.arange(X.shape[0])
    explainers = _make_shap_explainers(explanation_types, space_types, comparer, X, algorithm, masker)
    instance_names, X, X_display = _filter_instances_with_nonfinite_predictions(explainers, instance_names, X, X_display)
    explanations = VariousTypeExplanationsNamespace()
    _make_shap_values(explanations, X, X_display, explainers)
    _derive_shap_values(explanations, comparer, explanation_types)
    return _filter_nonfinite_shap_values(explanations, instance_names)


def _make_shap_explainers(explanation_types, space_types, comparer, X, algorithm, masker=None):
    if masker is None:
        masker = Independent(data=X)
    feature_names = comparer.feature_names
    combined_base_class_names = [f'A-{f}' for f in comparer.base_class_names] + \
                                [f'B-{f}' for f in comparer.base_class_names]

    explainers = SimpleNamespace()
    if 'indiv' in explanation_types:
        ns = SimpleNamespace()
        if 'labels' in space_types:
            ns.labels = _make_shap_explainer(comparer.predict_one_hot_combined, masker=masker,
                                             algorithm=algorithm, feature_names=feature_names,
                                             output_names=combined_base_class_names)
        if 'proba' in space_types:
            ns.proba = _make_shap_explainer(comparer.predict_proba_combined, masker=masker,
                                            algorithm=algorithm, feature_names=feature_names,
                                            output_names=combined_base_class_names)
        if 'log_odds' in space_types:
            ns.log_odds = _make_shap_explainer(comparer.predict_log_odds_combined, masker=masker,
                                               algorithm=algorithm, feature_names=feature_names,
                                               output_names=combined_base_class_names)
        explainers.indiv = ns
    if 'bin_diff' in explanation_types:
        ns = SimpleNamespace()
        if 'labels' in space_types:
            ns.labels = _make_shap_explainer(comparer.predict_bin_diff, masker=masker,
                                             algorithm=algorithm, feature_names=feature_names)
        if 'proba' in space_types:
            ns.proba = _make_shap_explainer(comparer.predict_bin_diff_proba, masker=masker,
                                            algorithm=algorithm, feature_names=feature_names)
        if 'log_odds' in space_types:
            ns.log_odds = _make_shap_explainer(comparer.predict_bin_diff_log_odds, masker=masker,
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
    for explanation_type, ns in explainers.__dict__.items():
        explainer = ns.__dict__.get('log_odds', None)
        if explainer is not None:
            y_pred = explainer.model(X)
            if len(y_pred.shape) == 1:
                mask = np.isfinite(y_pred)
            else:
                mask = np.all(np.isfinite(y_pred), axis=1)
            if np.sum(~mask) > 0:
                warnings.warn(f'Filtering instances with nonfinite predictions for {explanation_type}: {instance_names[~mask]}')
                filter_mask &= mask

    instance_names = instance_names[filter_mask]
    X = X[filter_mask, :]
    if X_display is not None:
        X_display = X_display[filter_mask, :]
    return instance_names, X, X_display


def _make_shap_values(explanations, X, X_display, explainers):
    for explanation_type, explainers_ns in explainers.__dict__.items():
        ns = SameTypeExplanationsNamespace()
        for space_type, explainer in explainers_ns.__dict__.items():
            shap_values = explainer(X)
            if X_display is not None:
                shap_values.display_data = X_display

            # workaround: restore output_names for explainers that don't support them
            if hasattr(explainer, 'output_names_backup'):
                shap_values = shap.Explanation(shap_values.values, shap_values.base_values, shap_values.data,
                                               shap_values.display_data, shap_values.instance_names,
                                               shap_values.feature_names, explainer.output_names_backup)
            ns.__dict__[space_type] = shap_values
        explanations.__dict__[explanation_type] = ns


def _derive_shap_values(explanations, comparer, explanation_types):
    if 'indiv_diff' in explanation_types:
        ns = SameTypeExplanationsNamespace()
        for space_type, base_shap_values in explanations.indiv.__dict__.items():
            shap_values_a = base_shap_values[:, :, :len(comparer.base_classes)]
            shap_values_b = base_shap_values[:, :, len(comparer.base_classes):]
            values = shap_values_a.values - shap_values_b.values
            base_values = shap_values_a.base_values - shap_values_b.base_values
            shap_values = shap.Explanation(values, base_values, base_shap_values.data, base_shap_values.display_data,
                                           feature_names=base_shap_values.feature_names,
                                           output_names=comparer.base_class_names)
            ns.__dict__[space_type] = shap_values
        explanations.indiv_diff = ns


def _filter_nonfinite_shap_values(explanations, instance_names):
    filter_mask = np.repeat(True, explanations.shape[0])
    for explanation_type, ns in explanations.__dict__.items():
        for space_type, shap_values in ns.__dict__.items():
            if len(shap_values.shape) == 2:
                mask = np.all(np.isfinite(shap_values.values), axis=1)
            else:
                mask = np.all(np.all(np.isfinite(shap_values.values), axis=2), axis=1)
            if np.sum(~mask) > 0:
                warnings.warn(f'Filtering instances with nonfinite shap values for {explanation_type}.{space_type}: {instance_names[~mask]}')
                filter_mask &= mask
    return explanations[filter_mask]
