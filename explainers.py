from sklearn.decomposition import PCA
import shap
from shap.maskers import Independent
from shap.utils import hclust_ordering
import numpy as np
import pickle
from util import encode_one_hot, reduce_multiclass_proba_diff_shap_values, \
                 calc_binary_log_odds_from_log_proba, calc_log_odds_from_log_proba
import warnings
import functools
from copy import copy
from plots import plot_2d, plot_2d_mclass, plot_feature_importance_bar, plot_feature_importance_bar_mclass, \
                  plot_feature_importance_scatter, plot_feature_importance_scatter_mclass, \
                  plot_feature_effects, plot_feature_effects_mclass, plot_forces, plot_forces_mclass, \
                  plot_decision_boundary
from collections.abc import Iterable


def generate(comparer, X, X_display=None, masker=None,
        kinds=['indiv-labels', 'indiv-proba', 'indiv-log-odds',
               'indiv-diff-labels', 'indiv-diff-proba', 'indiv-diff-log-odds',
               'bin-diff-labels', 'bin-diff-proba', 'bin-diff-log-odds',
               'mclass-diff-labels', 'mclass-diff-proba', 'mclass-diff-log-odds'],
        **kwargs):
    """ Generate shap values for all given kinds

    X_display : data to use for plots
    masker : shap.maskers.Masker instance, default is Independent
    kwargs : additional arguments are passed to shap.Explainer
             e.g. algorithm='permutation'
    """
    instance_names = np.arange(X.shape[0])
    tasks = _make_tasks(kinds, comparer)
    instance_names, X, X_display = _filter_instances_with_nonfinite_predictions(tasks, instance_names, X, X_display)
    shap_values = _make_shap_values(tasks, comparer.feature_names, X, X_display, masker, **kwargs)
    
    is_pred_diff = comparer.predict_bin_diff(X)
    base_class_names = np.array([str(label) for label in comparer.base_classes])
    class_names = np.array([str(label) for label in comparer.class_tuples])
    explanation = Explanation(shap_values, comparer.feature_names, is_pred_diff, instance_names, base_class_names, class_names)
    return _filter_nonfinite_shap_values(explanation)

def _make_tasks( kinds, comparer):
    tasks = {}
    for kind in kinds:
        tasks.update(_make_predict_tuples(kind, comparer))
    return tasks

def _make_predict_tuples(kind, comparer):
    if kind == 'indiv-labels':
        n_classes = len(comparer.base_classes)
        functions = [lambda X: encode_one_hot(f(X), n_classes) for f in comparer.predict_functions]
        return zip([kind + '-a', kind + '-b'], functions)
    if kind == 'indiv-proba':
        return zip([kind + '-a', kind + '-b'], comparer.predict_proba_functions)
    if kind == 'indiv-log-odds':
        return zip([kind + '-a', kind + '-b'], comparer.predict_log_odds_functions)
    if kind == 'indiv-diff-labels':
        n_classes = len(comparer.base_classes)
        return [(kind, lambda X: np.subtract(*[encode_one_hot(f(X), n_classes) for f in comparer.predict_functions]))]
    if kind == 'indiv-diff-proba':
        return [(kind, lambda X: np.subtract(*[f(X) for f in comparer.predict_proba_functions]))]
    if kind == 'indiv-diff-log-odds':
        return [(kind, lambda X: np.subtract(*[f(X) for f in comparer.predict_log_odds_functions]))]
    if kind == 'bin-diff-labels':
        return [(kind, comparer.predict_bin_diff)]
    if kind == 'bin-diff-proba':
        return [(kind, comparer.predict_bin_diff_proba)]
    if kind == 'bin-diff-log-odds':
        return [(kind, comparer.predict_bin_diff_log_odds)]
    if kind == 'mclass-diff-labels':
        n_classes = len(comparer.classes)
        f = lambda X: encode_one_hot(comparer.predict_mclass_diff(X), n_classes)
        return [(kind, f)]
    if kind == 'mclass-diff-proba':
        return [(kind, comparer.predict_mclass_diff_proba)]
    if kind == 'mclass-diff-log-odds':
        return [(kind, comparer.predict_mclass_diff_log_odds)]
    raise Exception(f'unsupported kind: {kind}')

def _filter_instances_with_nonfinite_predictions(tasks, instance_names, X, X_display=None):
    filter_mask = np.repeat(True, X.shape[0])
    for key, predict in tasks.items():
        if 'log-odds' in key:
            y_pred = predict(X)
            if len(y_pred.shape) == 1:
                mask = np.isfinite(y_pred)
            else:
                mask = np.all(np.isfinite(y_pred), axis=1)
            if np.sum(~mask) > 0:
                warnings.warn(f'Filtering instances with nonfinite predictions for {key}: {instance_names[~indices]}')
                filter_mask &= mask
                # TODO: save them for investigation

    instance_names = instance_names[filter_mask]
    X = X[filter_mask, :]
    if X_display is not None:
        X_display = X_display[filter_mask, :]
    return instance_names, X, X_display

def _make_shap_values(tasks, feature_names, X, X_display=None, masker=None, **kwargs):
    shap_values = {}
    for key, predict in tasks.items():
        shap_values[key] = _generate_shap_values(predict, feature_names, X, X_display, masker, **kwargs)
    return shap_values

def _generate_shap_values(f, feature_names, X, X_display=None, masker=None, **kwargs):
    """ We don't use shap.Explanation.output_names and output_indexes because afterwards slicing
    doesn't work anymore. Instead we keep our own in class_names, class_indices
    and base_class_names, base_class_indices
    """
    if masker is None:
        masker = Independent(data=X)

    explainer = shap.Explainer(f, masker, feature_names=feature_names, **kwargs)
    shap_values = explainer(X)
    shap_values.display_data = X_display
    return shap_values

def _filter_nonfinite_shap_values(explanation):
    filter_mask = np.repeat(True, explanation.shape[0])
    for key, shap_values in explanation.shap_values.items():
        if len(shap_values.shape) == 2:
            mask = np.all(np.isfinite(shap_values.values), axis=1)
        else:
            mask = np.all(np.all(np.isfinite(shap_values.values), axis=2), axis=1)
        if np.sum(~mask) > 0:
            warnings.warn(f'Filtering instances with nonfinite shap values for {key}: {explanation.instance_names[~mask]}')
            filter_mask &= mask
            # TODO: save them for investigation

    return explanation[filter_mask]


class Explanation:

    def __init__(self, shap_values, feature_names, is_pred_diff, instance_names, base_class_names, class_names,
                 feature_order=None, class_importances=None, class_order=None, n_informative_classes=None,
                 instance_order=None):
        if feature_order is None:
            feature_order = np.arange(len(feature_names))
        if class_importances is None:
            class_importances = np.repeat(1.0/len(class_names), len(class_names))
        if class_order is None:
            class_order = np.arange(len(class_names))
        if n_informative_classes is None:
            n_informative_classes = len(class_names)
        
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.feature_order = feature_order
        self.class_importances = class_importances
        self.class_order = class_order
        self.n_informative_classes = n_informative_classes
        self.instance_names = instance_names
        self.instance_order = instance_order
        self.is_pred_diff = is_pred_diff
        self.base_class_names = base_class_names
        self.base_class_indices = np.arange(len(self.base_class_names))
        self.class_names = class_names
        self.class_indices = np.arange(len(self.class_names))
        
    def __len__(self):
        return self.shape[0]
    
    @property
    def shape(self):
        return len(self.instance_names), len(self.feature_names), len(self.base_class_indices), len(self.class_indices)
    
    @property
    def informative_class_indices(self):
        return self.class_order[:self.n_informative_classes]

    def sample(self, max_samples, replace=False, random_state=None):
        """ Randomly samples the instances (rows) of the Explainer object.
        
        Parameters
        ----------
        max_samples : int
            The number of rows to sample. Note that if replace=False then less than
            fewer than max_samples will be drawn if explanation.shape[0] < max_samples.
        
        replace : bool
            Sample with or without replacement.
        
        random_state : bool
            Initialization seed for the random number generator.
        """
        rng = np.random.default_rng(random_state)
        indices = rng.choice(self.shape[0], min(max_samples, self.shape[0]), replace=replace)
        return self[list(indices)]
            
    def __getitem__(self, key):
        instance_order = self.instance_order
        feature_names = self.feature_names
        feature_order = self.feature_order
        class_importances = self.class_importances
        class_order = self.class_order
        n_informative_classes = self.n_informative_classes
        base_class_names = self.base_class_names
        class_names = self.class_names
        
        instance_key = None
        feature_key = None
        base_class_key = None
        diff_class_key = None

        if isinstance(key, tuple):
            instance_key = key[0]
            if len(key) >= 2:
                feature_key = key[1]
                feature_names = self.feature_names[feature_key]
                feature_order = self.feature_order[feature_key]
                feature_order = np.searchsorted(np.sort(feature_order), feature_order)
            if len(key) >= 3:
                base_class_key = key[2]
                base_class_names = self.base_class_names[base_class_key]
            if len(key) >= 4:
                diff_class_key = key[3]
                class_importances = self.class_importances[diff_class_key]
                class_order = self.class_order[diff_class_key]
                class_order = np.searchsorted(np.sort(class_order), class_order)
                is_informative_class = np.concatenate((np.repeat(True, self.n_informative_classes),
                                                       np.repeat(False, len(self.class_names) - self.n_informative_classes)))
                n_informative_classes = np.sum(is_informative_class[diff_class_key])
                class_names = self.class_names[diff_class_key]
        else:
            instance_key = key
           
        instance_names = self.instance_names[instance_key]
        is_pred_diff = self.is_pred_diff[instance_key]
        if self.instance_order is not None:
            instance_order = self.instance_order[instance_key]
            instance_order = np.searchsorted(np.sort(instance_order), instance_order)

        new_shap_values = {}
        for kind, shap_values in self.shap_values.items():
            if kind in ['indiv-labels', 'indiv-proba', 'indiv-log-odds']:
                continue
            
            if kind.startswith('indiv-'):
                current_key = instance_key, feature_key, base_class_key
            elif kind.startswith('bin-diff-'):
                current_key = instance_key, feature_key
            elif kind.startswith('mclass-diff-'):
                current_key = instance_key, feature_key, diff_class_key
            new_shap_values[kind] = shap_values[current_key]
            
        for key in self.shap_values.keys():
            if key.startswith('indiv-') and key.endswith('-a'):
                a = new_shap_values[key]
                b = new_shap_values[key[:-1] + 'b']
                values = np.stack((a.values, b.values), axis=2)
                if len(values.shape) > 3:
                    values = values.reshape((a.shape[0], a.shape[1], a.shape[2] + b.shape[2]))
                base_values = np.hstack((a.base_values, b.base_values))
                if len(base_values.shape) == 1:
                    base_values = base_values.reshape((a.shape[0], 2))
                s = shap.Explanation(values, base_values, a.data, a.display_data, a.instance_names, a.feature_names)
                new_shap_values[key[:-2]] = s

        return Explanation(new_shap_values, feature_names, is_pred_diff, instance_names,
                           base_class_names, class_names, feature_order, class_importances,
                           class_order, n_informative_classes, instance_order)
        
    def calc_feature_order(self, kind='bin-diff-labels'):
        shap_values = self.shap_values[kind]
        if len(shap_values.shape) == 3:
            shap_values = shap_values.abs.mean(axis=2)
        feature_importance = shap_values.abs.mean(axis=0)
        self.feature_order = np.flip(feature_importance.values.argsort())
        title = self._make_title(kind)
        plot_feature_importance_bar(feature_importance, title, self.feature_order)
        
    def calc_class_order(self, kind='mclass-diff-labels', information_threshold_pct=0.9):
        if not kind.startswith('mclass-'):
            raise Exception('only multiclass kinds allowed')
        shap_values = self.shap_values[kind]
        self.class_importances = np.abs(self.shap_values['mclass-diff-labels'].values).mean(axis=1).mean(axis=0)
        self.class_order = np.flip(np.argsort(self.class_importances))
        class_importances_cumulated = np.cumsum(self.class_importances[class_order])
        total_importance = np.sum(self.class_importances)
        proportional_importances = class_importances_cumulated / total_importance
        self.n_informative_classes = 1 + np.where(proportional_importances > information_threshold_pct)[0][0]
        # TODO: plot results
        
    def calc_instance_order(self, kind='bin-diff-labels'):
        values = self.shap_values[kind].values
        if len(values.shape) == 3:
            values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
        self.instance_order = np.argsort(hclust_ordering(values))

    def plot_2d(self, kind='mclass-diff-labels', idx_x=None, idx_y=None, **kwargs):
        if idx_x is None or idx_y is None:
            idx_x, idx_y = self.feature_order[:2]
        shap_values = self.shap_values[kind]
        title = self._make_title(kind)
        if len(shap_values.shape) == 2:
            return plot_2d(shap_values, title, idx_x, idx_y, self.feature_order, **kwargs)
        if len(shap_values.shape) == 3:
            class_indices, class_names = self._get_classes(kind)
            return plot_2d_mclass(shap_values, title, idx_x, idx_y, self.feature_order, class_indices, class_names, **kwargs)
        raise Exception(f'invalid shape: {shap_values.shape}')
        
    def _get_classes(self, kind):
        if kind in ['indiv-labels-a', 'indiv-labels-b',
                    'indiv-proba-a', 'indiv-proba-b',
                    'indiv-log-odds-a', 'indiv-log-odds-b',
                    'indiv-diff-labels', 'indiv-diff-proba', 'indiv-diff-log-odds']:
            return self.base_class_indices, self.base_class_names
        if kind in ['indiv-labels', 'indiv-proba', 'indiv-log-odds']:
            base_class_names = self.base_class_names
            if not isinstance(base_class_names, Iterable):
                base_class_names = [base_class_names]
            class_names = np.array([f'{f} of A' for f in base_class_names] + [f'{f} of B' for f in base_class_names])
            class_indices = np.arange(class_names.shape[0])
            return class_indices, class_names
        if kind.startswith('mclass-diff-'):
            return self.informative_class_indices, self.class_names
        
    def _make_title(self, kind):
        if kind.startswith('bin-diff-'):
            title = 'binary difference classifier'
        elif kind.startswith('indiv-'):
            if kind.startswith('indiv-diff-'):
                title = 'A-B'
            elif kind.endswith('-a'):
                title = 'A'
            elif kind.endswith('-b'):
                title = 'B'
            else:
                title = 'A and B'
                
            if isinstance(self.class_names, str):
                title += f' {self.class_names}'
        elif kind.startswith('mclass-diff-'):
            title = 'multiclass difference classifier'
            if isinstance(self.class_names, str):
                title += f' {self.class_names}'
                
        if len(self.instance_names.shape) == 0:
            title += f' #{self.instance_names}'
        return title
                
    def plot_feature_importance_bar(self, kind='bin-diff-labels'):
        shap_values = self.shap_values[kind]
        title = self._make_title(kind)
        if len(shap_values.shape) <= 2:
            return plot_feature_importance_bar(shap_values, title, self.feature_order)
        elif len(shap_values.shape) == 3:
            class_indices, class_names = self._get_classes(kind)
            return plot_feature_importance_bar_mclass(shap_values, title, class_indices, class_names)
        raise Exception(f'invalid shape: {shap_values.shape}')
        
    def plot_feature_importance_scatter(self, kind='bin-diff-labels'):
        shap_values = self.shap_values[kind]
        title = self._make_title(kind)
        if len(shap_values.shape) == 2:
            return plot_feature_importance_scatter(shap_values, title, self.feature_order)
        elif len(shap_values.shape) == 3:
            class_indices, class_names = self._get_classes(kind)
            return plot_feature_importance_scatter_mclass(shap_values, title, self.feature_order, class_indices, class_names)
        raise Exception(f'invalid shape: {shap_values.shape}')
                
    def plot_feature_effects(self, kind='bin-diff-labels', **kwargs):
        """Plot marginal effect of each feature vs. its SHAP values per class.
        
        Further keyword arguments are passed to shap.plots.scatter,
        and may include e.g. alpha=0.2
        """
        shap_values = self.shap_values[kind]
        title = self._make_title(kind)
        if len(shap_values.shape) == 2:
            return plot_feature_effects(shap_values, title, self.is_pred_diff, self.feature_order, **kwargs)
        elif len(shap_values.shape) == 3:
            class_indices, class_names = self._get_classes(kind)
            return plot_feature_effects_mclass(shap_values, title, self.is_pred_diff, self.feature_order, class_indices, class_names, **kwargs)
        raise Exception(f'invalid shape: {shap_values.shape}')
    
    def plot_forces(self, kind='bin-diff-labels', **kwargs):
        """Create force plots of all instances
        
        Further keyword arguments are passed to shap plot function
        e.g. link='logit'
        """
        shap_values = self.shap_values[kind]
        if shap_values.shape[0] > 1000:
            warnings.warn('plotting more than 1000 instances could slow down your browser. try sclicing or sampling')
        title = self._make_title(kind)
        if len(shap_values.shape) <= 2:
            return plot_forces(shap_values, title, self.instance_order, **kwargs)
        if len(shap_values.shape) == 3:
            class_indices, class_names = self._get_classes(kind)
            return plot_forces_mclass(shap_values, title, class_indices, class_names, self.instance_order, **kwargs)
        raise Exception(f'invalid shape: {shap_values.shape}')
                    
    def plot_heatmap(self, kind='mclass-diff-labels'):
        raise Exception(f'invalid shape: {shap_values.shape}')
    
    def plot_decision(self, kind='mclass-diff-labels'):
        raise Exception(f'invalid shape: {shap_values.shape}')