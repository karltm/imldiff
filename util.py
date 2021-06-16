import numpy as np
from scipy.special import logsumexp
import shap
from shap.utils import hclust_ordering
from sklearn.decomposition import PCA
    
def reduce_multiclass_proba_diff_shap_values(shap_values):
    """
    Reduce multiclass difference SHAP-values to binary class difference SHAP-values
    assuming they where generated on probability (or 0/1) estimates of the base models
    """
    difference_classes = get_difference_classes(n_classes=shap_values.values.shape[2])
    new_values = np.sum(shap_values.values[:, :, difference_classes], axis=2)
    new_base_values = np.sum(shap_values.base_values[:, difference_classes], axis=1)
    output_names = None
    output_indexes = None
    return shap.Explanation(new_values, new_base_values,
                            shap_values.data,
                            shap_values.display_data,
                            shap_values.instance_names,
                            shap_values.feature_names,
                            output_names,
                            output_indexes,
                            shap_values.main_effects,
                            shap_values.hierarchical_values,
                            shap_values.clustering)

def get_difference_classes(n_classes):
    classes = np.arange(n_classes)
    equality_classes = get_equality_classes(n_classes)
    return np.setdiff1d(classes, equality_classes)

def get_equality_classes(n_classes):
    n_base_classes = np.sqrt(n_classes).astype(int)
    class_matrix = np.arange(n_classes).reshape((n_base_classes, n_base_classes))
    return np.diagonal(class_matrix)

def encode_one_hot(labels, classes):
    indices = np.searchsorted(classes, labels)
    return np.eye(len(classes))[indices]

def calc_binary_log_odds_from_log_proba(log_proba):
    return log_proba[:, 1] - log_proba[:, 0]

def calc_log_odds_from_log_proba(log_proba):
    log_odds = np.empty(log_proba.shape)
    for i in range(log_proba.shape[1]):
        class_mask = [True] * log_proba.shape[1]
        class_mask[i] = False
        log_odds[:, i] = log_proba[:, i] - logsumexp(log_proba[:, class_mask], axis=1)
    return log_odds


def calc_feature_order(shap_values):
    if len(shap_values.shape) == 2:
        values = np.abs(shap_values.values).mean(axis=0)
    elif len(shap_values.shape) == 3:
        values = np.abs(shap_values.values).mean(axis=2).mean(axis=0)
    else:
        raise Exception(f'invalid dimensions: {shap_values.shape}')
    feature_order = np.flip(values.argsort())
    feature_importance = shap.Explanation(values, feature_names=shap_values.feature_names)
    return feature_order, feature_importance


def calc_class_order(shap_values, information_threshold_pct=0.9):
    if not len(shap_values.shape) == 3:
        raise Exception('only multiclass kinds allowed')
    class_importances = np.abs(shap_values.values).mean(axis=1).mean(axis=0)
    class_order = np.flip(np.argsort(class_importances))
    class_importances_cumulated = np.cumsum(class_importances[class_order])
    total_importance = np.sum(class_importances)
    proportional_importances = class_importances_cumulated / total_importance
    n_informative_classes = 1 + np.where(proportional_importances > information_threshold_pct)[0][0]
    return class_order, class_importances, n_informative_classes
    # TODO: plot results


def calc_instance_order(shap_values):
    values = shap_values.values
    if len(values.shape) == 3:
        values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    instance_order = np.argsort(hclust_ordering(values))
    return instance_order
    # informative_class_indices = class_order[:n_informative_classes]
