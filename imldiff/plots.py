import copy
import matplotlib.pyplot as plt
from IPython.core.display import display
import numpy as np
import shap
from sklearn.decomposition import PCA

color_map = copy.copy(plt.cm.get_cmap('RdBu').reversed())
color_map.set_under('blue')
color_map.set_over('red')
color_map_bright = copy.copy(plt.cm.get_cmap('bwr'))
color_map_bright.set_under('darkblue')
color_map_bright.set_under('darkred')


def plot_functions(X, functions, title, xlabel, ylabel, xlim=None, ylim=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if callable(functions):
        ax.plot(X[:,0], functions(X))
    elif isinstance(functions, dict):
        for label, func in functions.items():
            ax.plot(X[:,0], func(X), label=label)
        ax.legend()
    ax.grid()
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_decision_boundary(model, X, title=None, feature_names=None, zlim=None, fig=None, ax=None):
    h = .01  # step size in the mesh

    if zlim is not None:
        z_from, z_to = zlim
        levels = np.linspace(z_from, z_to, 21)
    else:
        z_from = None
        z_to = None
        levels = 21

    if not fig or not ax:
        fig = plt.figure(figsize=(9, 7))
        ax = plt.subplot()

    z = model(X)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = model(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cs = ax.contourf(xx, yy, Z, levels, cmap=color_map, alpha=.8)
    fig.colorbar(cs, ax=ax, shrink=0.9)

    ax.contourf(xx, yy, np.where(np.isposinf(Z), 1, None), colors='pink')
    ax.contourf(xx, yy, np.where(np.isneginf(Z), 1, None), colors='turquoise')
    ax.contourf(xx, yy, np.where(np.isnan(Z), 1, None), colors='lightyellow')

    # Plot the points
    ax.scatter(X[:, 0], X[:, 1], c=z, cmap=color_map_bright, vmin=z_from, vmax=z_to, edgecolors='k')
    ax.scatter(X[np.isneginf(z),0], X[np.isneginf(z),1], c='blue')
    ax.scatter(X[np.isposinf(z),0], X[np.isposinf(z),1], c='red')
    ax.scatter(X[np.isnan(z),0], X[np.isnan(z),1], c='yellow')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)


def get_display_range(model):
    if model.is_log_output_space():
        return -10, 10
    else:
        return 0, 1


def plot_shap_value_distribution(shap_values, title, xlim=None):
    shap.plots.beeswarm(shap_values, show=False, plot_size=(14, 7))
    plt.title(title)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    plt.show()


def plot_shap_partial_dependence(shap_values, feature, title, ylim=None, ax=None):
    ymin = None
    ymax = None
    if ylim is not None:
        ymin, ymax = ylim
    shap.plots.scatter(
        shap_values[:, feature],
        color=shap_values,
        title=title,
        ymin=ymin,
        ymax=ymax,
        ax=ax,
        show=False if ax else True)


def plot_shap_values_stacked(shap_values, title, ordering=None):
    plot = shap.plots.force(
        base_value=0.0,
        shap_values=shap_values.values,
        features=shap_values.display_data,
        feature_names=shap_values.feature_names,
        out_names=title,
        ordering_keys=ordering)
    display(plot)
    ordering = _get_force_plot_ordering(plot)
    return ordering


def _get_force_plot_ordering(plot):
    return list(map(lambda x: int(x['simIndex']), plot.data['explanations']))


def make_pca_embedding_values(shap_values):
    pca = PCA(2)
    return pca.fit_transform(shap_values.values)


def plot_shap_values_hierarchically_clustered(shap_values, title):
    shap.plots.heatmap(shap_values, max_display=shap_values.shape[1], show=False)
    plt.gcf().set_size_inches(7, 7)
    plt.title(title)
    plt.show()
