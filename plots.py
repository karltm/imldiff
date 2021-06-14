import matplotlib.pyplot as plt
from shap.plots import colors
import numpy as np
import shap

plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_decision_boundary(X, z=None, title=None, feature_names=None, X_display=None, predict=None,
                           idx_x=0, idx_y=1, class_names=None, zlim=None,
                           fig=None, ax=None, **kwargs):
    """
    - X: instances to plot
    - z: color of instances
    - title: for figure
    - feature_names
    - predict: draw contours for this function (only if number of features is 2)
    - idx_x: index of feature to plot on x axis
    - idx_y: index of feature to plot on y axis
    - class_names: set this to a list of class names if predict returns labels
    - zlim: set this to the range of values if predict returns a continuous variable, e.g. (0, 1)
    - fig, ax
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        
    if class_names is None and zlim is None:
        if z.dtype == int or z.dtype == bool:
            class_names = np.unique(z)
        else:
            zlim = np.min(z), np.max(z)
            
    if X_display is None:
        X_display = X
        
    if z is None:
        z = predict(X)
        
    draw_contours = predict is not None and X.shape[1] == 2
    if draw_contours:
        mesh_step_size = .01    
        x_min, x_max = X[:, idx_x].min() - .5, X[:, idx_x].max() + .5
        y_min, y_max = X[:, idx_y].min() - .5, X[:, idx_y].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))
        z_pred = predict(np.c_[xx.ravel(), yy.ravel()])
        z_pred = z_pred.reshape(xx.shape)
    
    if class_names is not None:
        if draw_contours:
            levels = np.arange(len(class_names) + 1)
            cs = ax.contourf(xx, yy, z_pred + 0.5, levels, colors=plt_colors, alpha=.8)
        for class_idx, class_ in enumerate(class_names):
            X_ = X_display[z == class_idx, :]
            if X_.shape[0] == 0:
                continue
            ax.scatter(X_[:, idx_x], X_[:, idx_y], c=plt_colors[class_idx], edgecolors='k', label=str(class_), **kwargs)
        ax.legend()
    else:
        if draw_contours:
            levels = np.linspace(zlim[0], zlim[1], 21)
            cs = ax.contourf(xx, yy, z_pred, levels, cmap=colors.red_blue, alpha=.8)
            fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.scatter(X[:, idx_x], X[:, idx_y], c=z, cmap=colors.red_blue, vmin=zlim[0], vmax=zlim[1], edgecolors='k', **kwargs)
 
    if feature_names is not None:
        ax.set_xlabel(feature_names[idx_x])
        ax.set_ylabel(feature_names[idx_y])
    ax.set_title(title)

def plot_2d(shap_values, title, idx_x=0, idx_y=1, feature_order=None, **kwargs):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    ncols = 1
    nrows = shap_values.shape[1]
    fig, axs = plt.subplots(nrows, ncols, figsize=(9*ncols, 9*nrows), constrained_layout=True)
    plot_idx = 0
    fig.suptitle('SHAP values of ' + title, fontsize=16)
    for feature_idx in feature_order:
        vmax = np.max(np.abs(shap_values.values[:, feature_idx]))
        ax = axs.flat[plot_idx]
        X_display = get_display_data(shap_values)
        cs = ax.scatter(X_display[:, idx_x],
                        X_display[:, idx_y],
                        c=shap_values.values[:, feature_idx],
                        vmin=-vmax, vmax=vmax,
                        cmap=colors.red_blue,
                        **kwargs)
        ax.set_title(shap_values.feature_names[feature_idx])
        ax.set_xlabel(shap_values.feature_names[idx_x])
        ax.set_ylabel(shap_values.feature_names[idx_y])
        plot_idx += 1
        fig.colorbar(cs, ax=ax, shrink=0.9)
    plt.show()

def get_display_data(shap_values):
    if shap_values.display_data is not None:
        return shap_values.display_data
    else:
        return shap_values.data
    
def plot_2d_mclass(shap_values, title, idx_x=0, idx_y=1, feature_order=None, class_order=None, class_names=None, **kwargs):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    if class_order is None:
        class_order = range(shap_values.shape[2])
    if class_names is None:
        class_names = [str(i) for i in class_order]
    ncols = shap_values.shape[2]
    nrows = shap_values.shape[1]
    fig, axs = plt.subplots(nrows, ncols, figsize=(9*ncols, 9*nrows), constrained_layout=True)
    plot_idx = 0
    fig.suptitle('SHAP values of ' + title, fontsize=16)
    for feature_idx in feature_order:
        vmax = np.max(np.abs(shap_values.values[:, feature_idx, :]))
        for class_idx in class_order:
            ax = axs.flat[plot_idx]
            X_display = get_display_data(shap_values)
            cs = ax.scatter(X_display[:, idx_x],
                            X_display[:, idx_y],
                            c=shap_values.values[:, feature_idx, class_idx],
                            vmin=-vmax, vmax=vmax,
                            cmap=colors.red_blue,
                            **kwargs)
            ax.set_title(f'SHAP-values of {shap_values.feature_names[feature_idx]} for {class_names[class_idx]}')
            ax.set_xlabel(shap_values.feature_names[idx_x])
            ax.set_ylabel(shap_values.feature_names[idx_y])
            plot_idx += 1
        fig.colorbar(cs, ax=ax, shrink=0.9)
    plt.show()
        
def plot_feature_importance_bar(shap_values, title, feature_order=None):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    plt.title('SHAP feature importance of ' + title)
    shap.plots.bar(shap_values, order=feature_order, max_display=len(feature_order))
    
def plot_feature_importance_bar_mclass(shap_values, title, class_order=None, class_names=None):
    if class_order is None:
        class_order = range(shap_values.shape[2])
    if class_names is None:
        class_names = [str(i) for i in class_order]
    shap_values_list = [values.T for values in shap_values.values.T]
    shap.summary_plot(shap_values_list, shap_values.data,
                      feature_names=shap_values.feature_names,
                      class_names=class_names, show=False)
    plt.legend(loc='right')
    plt.title('SHAP feature importance of ' + title)
    plt.show()
    
def plot_feature_importance_scatter(shap_values, title, feature_order=None):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    plt.title('SHAP values of ' + title)
    shap.plots.beeswarm(shap_values, order=feature_order, plot_size=(14, 7))

def plot_feature_importance_scatter_mclass(shap_values, title, feature_order=None, class_order=None, class_names=None):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    if class_order is None:
        class_order = range(shap_values.shape[2])
    if class_names is None:
        class_names = [str(i) for i in class_order]
    plt.suptitle('SHAP values of ' + title, fontsize='x-large')
    for feature_idx in feature_order:
        new_values = shap_values.values[:, feature_idx, :]
        new_data = np.reshape(np.repeat(shap_values.data[:, feature_idx], len(class_order)),
                              (shap_values.data.shape[0], len(class_order)))
        if shap_values.display_data is not None:
            new_display_data = np.reshape(np.repeat(shap_values.display_data[:, feature_idx], len(class_order)),
                                          (shap_values.data.shape[0], len(class_order)))
        else:
            new_display_data = None
        new_base_values = shap_values.base_values
        shap_values_ = shap.Explanation(new_values, new_base_values, new_data, new_display_data, feature_names=class_names)
        shap.plots.beeswarm(shap_values_, plot_size=(14, 7), show=False)
        plt.title(shap_values.feature_names[feature_idx])
        plt.show()
        
def plot_feature_effects(shap_values, title, color=None, feature_order=None, **kwargs):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    ncols = 1
    nrows = shap_values.shape[1]
    fig = plt.figure(figsize=(9*ncols, 7*nrows))
    fig.suptitle('Feature effects of ' + title, fontsize='x-large')
    i = 1
    for feature_idx in feature_order:
        ax = fig.add_subplot(nrows, ncols, i)
        shap.plots.scatter(shap_values[:, feature_idx],
                           title=shap_values.feature_names[feature_idx],
                           ax=ax,
                           color=color,
                           show=False,
                           **kwargs)
        i += ncols
    plt.show()
    
    
def plot_feature_effects_mclass(shap_values, title, color=None, feature_order=None, class_order=None, class_names=None, **kwargs):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    if class_order is None:
        class_order = range(shap_values.shape[2])
    if class_names is None:
        class_names = [str(i) for i in class_order]
    ncols = len(class_order)
    nrows = shap_values.shape[1] * 2
    fig = plt.figure(figsize=(9 * ncols, 7 * nrows))
    fig.suptitle('Feature effects of ' + title, fontsize='x-large', y=0.91)
    plot_idx = 1
    for feature_idx in feature_order:
        ax_ref = None
        for class_idx in class_order:
            ax = fig.add_subplot(nrows, ncols, plot_idx, sharey=ax_ref)
            if not ax_ref:
                ax_ref = ax
            shap.plots.scatter(shap_values[:, feature_idx, class_idx],
                               title=class_names[class_idx],
                               color=color,
                               ax=ax, show=False, **kwargs)
            plot_idx += 1
    plt.show()
    
def plot_forces(shap_values, title, instance_order=None, **kwargs):
    if instance_order is not None and isinstance(instance_order, np.ndarray):
        instance_order = instance_order.tolist()
    X_display = get_display_data(shap_values)
    plot = shap.plots.force(
        base_value=shap_values.base_values[0],
        shap_values=shap_values.values,
        features=X_display,
        feature_names=shap_values.feature_names,
        out_names=title,
        ordering_keys=instance_order,
        **kwargs)
    display(plot)

def plot_forces_mclass(shap_values, title, class_order=None, class_names=None, instance_order=None, **kwargs):
    if class_order is None:
        class_order = range(shap_values.shape[2])
    if class_names is None:
        class_names = [str(i) for i in class_order]
    if instance_order is not None and isinstance(instance_order, np.ndarray):
        instance_order = instance_order.tolist()
    X_display = get_display_data(shap_values)
    for class_idx in class_order:
        shap_values_ = shap_values[:, :, class_idx]
        plot = shap.plots.force(
            base_value=shap_values_.base_values[0],
            shap_values=shap_values_.values,
            features=X_display,
            feature_names=shap_values.feature_names,
            out_names=str(class_names[class_idx]),
            ordering_keys=instance_order)
        display(plot)
