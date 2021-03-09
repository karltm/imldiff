import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import shap
from sklearn.decomposition import PCA


def plot_decision_boundaries(models, X, y=None):
    """plot low predicted values blue and high predicted values red"""
    figure = plt.figure(figsize=(27, 9))
    cm = plt.cm.RdBu.reversed()
    cm_bright = ListedColormap(['#0000FF', '#FF0000'])

    i = 1

    for model in models:
        ax = plt.subplot(1, len(models), i)

        h = .02  # step size in the mesh

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(model)

        i += 1


def force_plot(explainer, ordering=None):
    return shap.plots.force(
        base_value=explainer.shap_values.abs.mean(0).base_values,
        shap_values=explainer.shap_values.values,
        features=explainer.shap_values.display_data,
        feature_names=explainer.shap_values.feature_names,
        out_names=str(explainer),
        ordering_keys=ordering)


def get_force_plot_ordering(plot):
    return list(map(lambda x: int(x['simIndex']), plot.data['explanations']))


def make_pca_embedding_values(explainer):
    pca = PCA(2)
    return pca.fit_transform(explainer.shap_values.values)

