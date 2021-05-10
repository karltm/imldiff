import numpy as np
from scipy.special import logsumexp
import shap
    
def reduce_multiclass_proba_diff_shap_values(shap_values):
    """
    Reduce multiclass difference model SHAP-values to binary class difference SHAP-values
    assuming they where generated on probability estimates of the base models
    """
    n_diff_classes = shap_values.values.shape[2]
    n_base_classes = np.sqrt(n_diff_classes).astype(int)
    class_matrix = np.arange(n_diff_classes).reshape((n_base_classes, n_base_classes))
    equality_classes = np.diagonal(class_matrix)
    new_values = np.sum(shap_values.values[:, :, equality_classes], axis=2)
    new_base_values = np.sum(shap_values.base_values[:, equality_classes], axis=1)
    return shap.Explanation(new_values, new_base_values,
                            data=shap_values.data,
                            display_data=shap_values.display_data,
                            feature_names=shap_values.feature_names)

def make_predict_one_hot_encoded_labels(n_classes, predict):
    return lambda X: np.eye(n_classes)[predict(X)]

def calc_binary_log_odds_from_log_proba(log_proba):
    return log_proba[:, 1] - log_proba[:, 0]

def calc_log_odds_from_log_proba(log_proba):
    log_odds = np.empty(log_proba.shape)
    for i in range(log_proba.shape[1]):
        class_mask = [True] * log_proba.shape[1]
        class_mask[i] = False
        log_odds[:, i] = log_proba[:, i] - logsumexp(log_proba[:, class_mask], axis=1)
    return log_odds

def get_force_plot_ordering(plot):
    return list(map(lambda x: int(x['simIndex']), plot.data['explanations']))