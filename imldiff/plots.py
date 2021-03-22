import copy
import matplotlib.pyplot as plt
from IPython.core.display import display
import numpy as np
import shap
from matplotlib import ticker
from sklearn.decomposition import PCA
from imldiff.result import Result, merge_functions


figure_height = 7
figure_width = 7

color_map = copy.copy(plt.cm.get_cmap('RdBu').reversed())
color_map.set_under('blue')
color_map.set_over('red')
color_map_bright = copy.copy(plt.cm.get_cmap('bwr'))
color_map_bright.set_under('darkblue')
color_map_bright.set_under('darkred')


def plot_distribution(ax, x, y, *results, space='proba', xlim_from=-1, xlim_to=1):
    if space == 'log-odds':
        ylim_from, ylim_to = -10, 10
    else:
        ylim_from, ylim_to = 0, 1
    for i, result in enumerate(results, 1):
        ax.plot(x, result.values, label=f'y{i}')
    ax.grid()
    ax.set_xlim(xlim_from, xlim_to)
    ax.set_ylim(ylim_from, ylim_to)
    ax.set_xlabel('x')
    ax.set_title(' vs. '.join([str(result) for result in results]))
    ax.legend()


def plot_decision_boundaries(models, X, feature_names=None):
    num_figures = len(models)
    num_columns = 2
    num_rows = int(np.ceil(num_figures/num_columns))
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns*(figure_width+2), num_rows*figure_height))
    for model, ax in zip(models, axs.ravel()):
        plot_decision_boundary(model, X, feature_names, fig, ax)


def plot_decision_boundary(model, X, title=None, z_from=None, z_to=None, levels=None, feature_names=None, fig=None, ax=None):
    h = .01  # step size in the mesh

    if levels is None:
        if z_from is not None and z_to is not None:
            levels = np.linspace(z_from, z_to, 11)
        else:
            levels = 10

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

    # Plot the points
    ax.scatter(X[:, 0], X[:, 1], c=z, cmap=color_map_bright, vmin=z_from, vmax=z_to, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    return cs.levels


def get_display_range(model):
    if model.is_log_output_space():
        return -10, 10
    else:
        return 0, 1


def plot_shap_value_distribution(explanation):
    shap.plots.beeswarm(explanation.values, show=False)
    plt.title(str(explanation))
    plt.xlim((-1, 1))
    plt.show()


def plot_shap_partial_dependence(explanation):
    for feature_name in explanation.values.feature_names:
        shap_values = explanation.values[:, feature_name]
        shap.plots.scatter(shap_values, color=explanation.values, title=str(explanation), ymin=-1, ymax=1)


def plot_shap_values_stacked(*explanations):
    ordering = None
    for explanation in explanations:
        if not ordering:
            plot = shap_force_plot(explanation)
            ordering = get_force_plot_ordering(plot)
        else:
            plot = shap_force_plot(explanation, ordering)
        display(plot)


def shap_force_plot(explanation, ordering=None, link='identity'):
    return shap.plots.force(
        base_value=explanation.values.values.mean(),
        shap_values=explanation.values.values,
        features=explanation.values.display_data,
        feature_names=explanation.values.feature_names,
        out_names=str(explanation),
        ordering_keys=ordering,
        link=link)


def get_force_plot_ordering(plot):
    return list(map(lambda x: int(x['simIndex']), plot.data['explanations']))


def make_pca_embedding_values(explainer):
    pca = PCA(2)
    return pca.fit_transform(explainer.shap_values.values)


def plot_shap_values_hierarchically_clustered(explanation):
    shap.plots.heatmap(explanation.values, max_display=explanation.values.shape[1], show=False)
    plt.gcf().set_size_inches(figure_width, figure_height)
    plt.title(str(explanation))
    plt.show()
