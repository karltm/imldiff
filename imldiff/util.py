import numpy as np
import shap

def reduce_nclass_proba_shap_values(shap_values):
    """
    Reduce multiclass difference model SHAP-values to binary class diffference SHAP-values
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

def get_force_plot_ordering(plot):
    return list(map(lambda x: int(x['simIndex']), plot.data['explanations']))