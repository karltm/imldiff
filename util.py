import numpy as np
from scipy.special import logsumexp
import shap
    
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

def encode_one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]

def calc_binary_log_odds_from_log_proba(log_proba):
    return log_proba[:, 1] - log_proba[:, 0]

def calc_log_odds_from_log_proba(log_proba):
    log_odds = np.empty(log_proba.shape)
    for i in range(log_proba.shape[1]):
        class_mask = [True] * log_proba.shape[1]
        class_mask[i] = False
        log_odds[:, i] = log_proba[:, i] - logsumexp(log_proba[:, class_mask], axis=1)
    return log_odds
