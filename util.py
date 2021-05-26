import numpy as np
from scipy.special import logsumexp
import shap
from shap.plots import colors
import matplotlib.pyplot as plt
    
def reduce_multiclass_proba_diff_shap_values(shap_values):
    """
    Reduce multiclass difference model SHAP-values to binary class difference SHAP-values
    assuming they where generated on probability estimates of the base models
    """
    equality_classes = get_equality_classes(n_diff_classes=shap_values.values.shape[2])
    new_values = 1 - np.sum(shap_values.values[:, :, equality_classes], axis=2)
    new_base_values = np.sum(shap_values.base_values[:, equality_classes], axis=1)
    return shap.Explanation(new_values, new_base_values,
                            data=shap_values.data,
                            display_data=shap_values.display_data,
                            feature_names=shap_values.feature_names)

def get_equality_classes(n_diff_classes):
    n_base_classes = np.sqrt(n_diff_classes).astype(int)
    class_matrix = np.arange(n_diff_classes).reshape((n_base_classes, n_base_classes))
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

def get_force_plot_ordering(plot):
    return list(map(lambda x: int(x['simIndex']), plot.data['explanations']))


plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_decision_boundary(predict, data, z, title, feature_names, x_feature=0, y_feature=1, class_names=None, zlim=None, fig=None, ax=None):
    """
    2D plot of a predict function. Either class_names or zlim has to be passed.
    - predict: function that returns a 1D numpy array
    - data: instances to plot
    - z: color of instances
    - title: for figure
    - feature_names
    - x_feature: index of feature to plot on x axis
    - y_feature: index of feature to plot on y axis
    - class_names: set this to a list of class names if predict returns labels
    - zlim: set this to the range of values if predict returns a continuous variable, e.g. (0, 1)
    - fig, ax
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        
    mesh_step_size = .01
    x_min, x_max = data[:, x_feature].min() - .5, data[:, x_feature].max() + .5
    y_min, y_max = data[:, y_feature].min() - .5, data[:, y_feature].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))
    z_pred = predict(np.c_[xx.ravel(), yy.ravel()])
    z_pred = z_pred.reshape(xx.shape)
    
    if class_names is not None:
        levels = np.arange(len(class_names) + 1)
        cs = ax.contourf(xx, yy, z_pred + 0.5, levels, colors=plt_colors, alpha=.8)
        for class_idx, class_ in enumerate(class_names):
            data_ = data[z == class_idx, :]
            ax.scatter(data_[:, 0], data_[:, 1], c=plt_colors[class_idx], edgecolors='k', label=str(class_))
        ax.legend()
    else:
        levels = np.linspace(zlim[0], zlim[1], 21)
        cs = ax.contourf(xx, yy, z_pred, levels, cmap=colors.red_blue, alpha=.8)
        ax.scatter(data[:, 0], data[:, 1], c=z, cmap=colors.red_blue, vmin=zlim[0], vmax=zlim[1], edgecolors='k')
        fig.colorbar(cs, ax=ax, shrink=0.9)
 
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(feature_names[x_feature])
    ax.set_ylabel(feature_names[y_feature])
    ax.set_title(title)