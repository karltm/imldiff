import matplotlib.pyplot as plt
from IPython.core.display import display
import numpy as np
import shap
from sklearn.decomposition import PCA


color_map = plt.cm.RdBu.reversed()
color_map_bright = plt.cm.bwr


def plot_decision_boundaries(predict, X, y=None, ax=None, title=None):
    """plot low predicted values blue and high predicted values red"""
    if not ax:
        plt.figure(figsize=(9, 9))
        ax = plt.subplot()

    h = .02  # step size in the mesh

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = predict(np.c_[xx.ravel(), yy.ravel()])

    Z_filtered = Z[~np.isinf(Z) & ~np.isnan(Z)]
    if len(Z_filtered) > 0:
        min_value = Z_filtered.min() - 1
        max_value = Z_filtered.max() + 1
    else:
        min_value = 0
        max_value = 1

    Z[np.isneginf(Z)] = min_value
    Z[np.isposinf(Z)] = max_value

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=color_map, alpha=.8)

    # Plot the points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=color_map_bright, edgecolors='k')

    neginfs = np.isneginf(y)
    ax.scatter(X[neginfs, 0], X[neginfs, 1], c='c')

    posinfs = np.isposinf(y)
    ax.scatter(X[posinfs, 0], X[posinfs, 1], c='m')

    nans = np.isnan(y)
    ax.scatter(X[nans, 0], X[nans, 1], c='y')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)


def plot_decision_boundaries_and_distributions(model, X):
    num_figures_in_row = 2
    fig, axs = plt.subplots(2, num_figures_in_row, figsize=(num_figures_in_row * 9, 9 + 3), gridspec_kw={'height_ratios': [3, 1]})
    predict_functions = [model.predict_proba, model.predict_log_odds]
    titles = [f'{model}', f'Log Odds {model}']
    for i, (predict, title) in enumerate(zip(predict_functions, titles)):
        Z = predict(X)
        plot_decision_boundaries(predict, X, Z, axs[0, i], title)
        Z_filtered = filter_nans_and_infinities(Z, model)
        axs[1, i].hist(Z_filtered, bins=25, color='grey')


def filter_nans_and_infinities(Z, model):
    nans = np.isnan(Z)
    nan_count = np.count_nonzero(nans)
    posinfs = np.isposinf(Z)
    posinf_count = np.count_nonzero(posinfs)
    neginfs = np.isneginf(Z)
    neginf_count = np.count_nonzero(neginfs)
    if nan_count > 0 or posinf_count > 0 or neginf_count > 0:
        print(f'{model}: Filtering {nan_count} NaNs, {posinf_count} infinities, {neginf_count} negative infinities')
        return Z[~(nans | posinfs | neginfs)]
    return Z


def plot_shap_value_distribution(explainer):
    shap.plots.beeswarm(explainer.shap_values, show=False)
    plt.title(str(explainer))
    plt.show()


def plot_shap_partial_dependence(explainer):
    for feature_name in explainer.shap_values.feature_names:
        shap_values = explainer.shap_values[:, feature_name]
        shap.plots.scatter(shap_values, color=explainer.shap_values, title=str(explainer))


def plot_shap_values_stacked(*explainers):
    ordering = None
    for explainer in explainers:
        if not ordering:
            plot = shap_force_plot(explainer)
            ordering = get_force_plot_ordering(plot)
        else:
            plot = shap_force_plot(explainer, ordering)
        display(plot)


def shap_force_plot(explainer, ordering=None, link='identity'):
    return shap.plots.force(
        base_value=explainer.shap_values.abs.mean(0).base_values,
        shap_values=explainer.shap_values.values,
        features=explainer.shap_values.display_data,
        feature_names=explainer.shap_values.feature_names,
        out_names=str(explainer),
        ordering_keys=ordering,
        link=link)


def get_force_plot_ordering(plot):
    return list(map(lambda x: int(x['simIndex']), plot.data['explanations']))


def make_pca_embedding_values(explainer):
    pca = PCA(2)
    return pca.fit_transform(explainer.shap_values.values)


def plot_shap_values_hierarchically_clustered(explainer):
    shap.plots.heatmap(explainer.shap_values, max_display=explainer.shap_values.shape[1], show=False)
    plt.gcf().set_size_inches(9, 9)
    plt.title(str(explainer))
    plt.show()
