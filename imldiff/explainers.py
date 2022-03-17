from typing import Iterable
import shap
from shap.maskers import Independent
from shap.utils import hclust_ordering
from shap.plots import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from comparers import ModelComparer
from IPython.display import display
from shap.utils import approximate_interactions
from shap.plots.colors import red_blue


color_palette = plt.rcParams["axes.prop_cycle"].by_key()['color']


def generate_shap_explanations(comparer: ModelComparer, X: np.ndarray, X_display: np.ndarray = None,
                               explanation_type='mclass_diff', space_type=None,
                               algorithm='auto', masker: shap.maskers.Masker = None):
    """ Generate SHAP values for difference classifier

    :param comparer: model comparison helper
    :param X: dataset to generate SHAP values for
    :param X_display: dataset with same shape as X, which is used for plots
                      and may contain descriptive categorical values
    :param explanation_type:
                'indiv' - individual models' SHAP values
                'bin_diff' - SHAP values of binary difference classifier
                'mclass_diff' - SHAP values of multiclass difference classifier
    :param space_type:
                'labels' - predicted labels (hard decision boundary)
                'proba' - predicted probabilities (soft decision boundary)
                'log_odds' - predicted log odds (soft decision boundary)
    :param algorithm: SHAP value generation algorithm. See shap.Explainer for possible values
    :param masker: Specify, if you want to customize the masker used during SHAP value generation
    """
    if space_type is None:
        if comparer.has_log_odds_support:
            space_type = 'log_odds'
        elif comparer.has_probability_support:
            space_type = 'proba'
        else:
            space_type = 'labels'

    predict_func, class_names = _get_predict_function_and_class_names(comparer, explanation_type, space_type)

    instance_indices = np.arange(X.shape[0])
    if space_type == 'log_odds':
        # SHAP algorithm fails for infinite log odds predictions
        y_pred = predict_func(X)
        if len(y_pred.shape) == 1:
            mask = np.isfinite(y_pred)
        else:
            mask = np.all(np.isfinite(y_pred), axis=1)
        if np.sum(~mask) > 0:
            warnings.warn(f'filtering instances with nonfinite predictions: {instance_indices[~mask]}')
        instance_indices = instance_indices[mask]

    explainer = shap.Explainer(predict_func, masker=masker, algorithm=algorithm,
                               feature_names=comparer.feature_names, output_names=class_names)
    shap_values = explainer(X[instance_indices])
    if X_display is not None:
        shap_values.display_data = X_display[instance_indices]

    # workaround for bug, where output_names aren't set in some cases
    if shap_values.output_names is None:
        shap_values = shap.Explanation(shap_values.values, shap_values.base_values, shap_values.data,
                                       shap_values.display_data, shap_values.instance_names,
                                       shap_values.feature_names, class_names)

    if space_type == 'log_odds':
        # Visualizations fail for non-finite SHAP values
        if len(shap_values.shape) == 2:
            mask = np.all(np.isfinite(shap_values.values), axis=1)
        else:
            mask = np.all(np.all(np.isfinite(shap_values.values), axis=2), axis=1)
        if np.sum(~mask) > 0:
            warnings.warn(f'filtering instances with non-finite SHAP values: {instance_indices[~mask]}')
            shap_values = shap_values[mask]

    return shap_values


def _get_predict_function_and_class_names(comparer, explanation_type, space_type):
    if explanation_type == 'indiv':
        if space_type == 'labels':
            predict_func = comparer.predict_combined_oh_encoded
            class_names = comparer.combined_class_names
        elif space_type == 'proba':
            predict_func = comparer.predict_combined_proba
            class_names = comparer.combined_class_names
        elif space_type == 'log_odds':
            predict_func = comparer.predict_combined_log_odds
            class_names = comparer.combined_class_names
        else:
            raise Exception(f'Invalid space type: {space_type}')
    elif explanation_type == 'bin_diff':
        if space_type == 'labels':
            predict_func = comparer.predict_bin_diff
            class_names = comparer.bin_class_names[-1]
        elif space_type == 'proba':
            predict_func = comparer.predict_bin_diff_proba
            class_names = comparer.bin_class_names[-1]
        elif space_type == 'log_odds':
            predict_func = comparer.predict_bin_diff_log_odds
            class_names = comparer.bin_class_names[-1]
        else:
            raise Exception(f'Invalid space type: {space_type}')
    elif explanation_type == 'mclass_diff':
        if space_type == 'labels':
            predict_func = comparer.predict_mclass_diff_oh_encoded
            class_names = comparer.class_names
        elif space_type == 'proba':
            predict_func = comparer.predict_mclass_diff_proba
            class_names = comparer.class_names
        elif space_type == 'log_odds':
            predict_func = comparer.predict_mclass_diff_log_odds
            class_names = comparer.class_names
        else:
            raise Exception(f'Invalid space type: {space_type}')
    else:
        raise Exception(f'Invalid explanation type: {explanation_type}')
    return predict_func, class_names


def ensure_shap_values_are_3d(shap_values):
    if len(shap_values.shape) == 3:
        return shap_values
    if isinstance(shap_values.feature_names, str):
        feature_names = [shap_values.feature_names]
    else:
        feature_names = shap_values.feature_names
    if isinstance(shap_values.output_names, str):
        output_names = [shap_values.output_names]
    else:
        output_names = shap_values.output_names
    values = shap_values.values.reshape((shap_values.shape[0], len(feature_names), len(output_names)))
    base_values = shap_values.base_values.reshape((shap_values.shape[0], len(output_names)))
    return shap.Explanation(values, base_values, shap_values.data, shap_values.display_data,
                            feature_names=feature_names, output_names=output_names)


def ensure_all_shap_values_are_3d(*shap_values_tuple, **kw_shap_values):
    if len(shap_values_tuple) > 0:
        return tuple([ensure_shap_values_are_3d(s) for s in shap_values_tuple])
    else:
        return dict([(k, ensure_shap_values_are_3d(s)) for k, s in kw_shap_values.items()])


def calc_feature_order(shap_values):
    shap_values = ensure_shap_values_are_3d(shap_values)
    values = np.abs(shap_values.values).mean(axis=2).mean(axis=0)
    feature_order = np.flip(values.argsort())
    feature_importance = shap.Explanation(values, feature_names=shap_values.feature_names)
    return feature_order, feature_importance


def calc_class_order(shap_values):
    if not len(shap_values.shape) == 3:
        raise Exception('only multiclass kinds allowed')
    class_importances = np.abs(shap_values.values).mean(axis=1).mean(axis=0)
    class_order = np.flip(np.argsort(class_importances))
    return class_order, class_importances


def plot_class_importances(class_importances, class_order, class_names):
    df = pd.DataFrame(class_importances[class_order], index=np.array(class_names)[class_order])
    df.plot.bar(title='Class importances', ylabel='mean(|SHAP value|)')


def calc_instance_order(shap_values):
    shap_values = ensure_shap_values_are_3d(shap_values)
    values = shap_values.values.reshape(
        (shap_values.values.shape[0],
         shap_values.values.shape[1] * shap_values.values.shape[2]))
    instance_order = np.argsort(hclust_ordering(values))
    return instance_order


def estimate_feature_interaction_order(shap_values, feature):
    if len(shap_values.shape) > 2:
        raise Exception('only one target class supported')
    feature_names = np.array(shap_values.feature_names)
    if isinstance(feature, str):
        feature = np.where(feature_names == feature)[0][0]
    return approximate_interactions(feature, shap_values.values, shap_values.data)


def plot_data_2d(*shap_values_tuple, title=None, x=0, y=1, **kwargs):
    shap_values_tuple = ensure_all_shap_values_are_3d(*shap_values_tuple)
    ncols = sum([s.shape[2] for s in shap_values_tuple])
    nrows = shap_values_tuple[0].shape[1]
    fig, axs = plt.subplots(nrows, ncols, figsize=(9*ncols, 9*nrows), constrained_layout=True, sharex=True, sharey=True)
    plot_idx = 0
    fig.suptitle(title, fontsize=16)
    for feature_idx in range(nrows):
        vmax = np.max([np.abs(s[:, feature_idx, :].values).flatten().max(0) for s in shap_values_tuple])
        for s in shap_values_tuple:
            display_shap_values = s[:, [x, y], :]
            X_display = _get_display_data(display_shap_values)
            for class_idx in range(s.shape[2]):
                ax = axs.flat[plot_idx]
                cs = ax.scatter(X_display[:, 0],
                                X_display[:, 1],
                                c=s.values[:, feature_idx, class_idx],
                                vmin=-vmax, vmax=vmax,
                                cmap=colors.red_blue,
                                **kwargs)
                ax.set_title(f'SHAP-values of {s.feature_names[feature_idx]} '
                             f'for {s.output_names[class_idx]}')
                ax.set_xlabel(display_shap_values.feature_names[0])
                ax.set_ylabel(display_shap_values.feature_names[1])
                plot_idx += 1
        fig.colorbar(cs, ax=ax, shrink=0.9)


def _get_display_data(shap_values):
    if shap_values.display_data is not None:
        return shap_values.display_data
    else:
        return shap_values.data


def plot_feature_importance_bar(shap_values, title=None, feature_order=None):
    if len(shap_values.shape) <= 2:
        return _plot_feature_importance_bar_singleclass(shap_values, title, feature_order)
    elif len(shap_values.shape) == 3:
        return _plot_feature_importance_bar_multiclass(shap_values, title)
    raise Exception(f'invalid dimensions: {shap_values.shape}')


def _plot_feature_importance_bar_singleclass(shap_values, title=None, feature_order=None):
    if feature_order is None:
        if len(shap_values.shape) == 2:
            feature_order = range(shap_values.shape[1])
        elif len(shap_values.shape) == 1:
            feature_order = np.flip(np.argsort(shap_values.values))
    plt.title(title)
    shap.plots.bar(shap_values, order=feature_order, max_display=len(feature_order), show=False)


def _plot_feature_importance_bar_multiclass(shap_values, title=None):
    shap_values_list = [values.T for values in shap_values.values.T]
    shap.summary_plot(shap_values_list, shap_values.data,
                      feature_names=shap_values.feature_names,
                      class_names=shap_values.output_names, show=False)
    plt.legend(loc='right')
    plt.title(title)


def plot_feature_importance_scatter(shap_values, title=None, feature_order=None, class_order=None, **kwargs):
    if shap_values.output_names is None or isinstance(shap_values.output_names, str):
        return _plot_feature_importance_scatter_singleclass(shap_values, title, feature_order, **kwargs)
    else:
        return _plot_feature_importance_scatter_multiclass(shap_values, title, feature_order, class_order, **kwargs)


def _plot_feature_importance_scatter_singleclass(shap_values, title=None, feature_order=None, plot_size=None, **kwargs):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    if plot_size is None:
        plot_size = (18, 3)
    plt.title(title)
    shap.plots.beeswarm(shap_values, order=feature_order, plot_size=plot_size, **kwargs)


def _plot_feature_importance_scatter_multiclass(shap_values, title=None, feature_order=None, class_order=None,
                                                plot_size=None, **kwargs):
    if feature_order is None:
        feature_order = range(shap_values.shape[1])
    if class_order is None:
        class_order = np.arange(shap_values.shape[2])
    if plot_size is None:
        plot_size = (18, 1 + shap_values.shape[2])
    plt.suptitle(title, fontsize='x-large')
    for feature_idx in feature_order:
        new_values = shap_values.values[:, feature_idx, :]
        new_data = np.reshape(np.repeat(shap_values.data[:, feature_idx], shap_values.shape[2]),
                              (shap_values.data.shape[0], shap_values.shape[2]))
        if shap_values.display_data is not None:
            new_display_data = np.reshape(np.repeat(shap_values.display_data[:, feature_idx], shap_values.shape[2]),
                                          (shap_values.data.shape[0], shap_values.shape[2]))
        else:
            new_display_data = None
        new_base_values = shap_values.base_values
        shap_values_ = shap.Explanation(new_values, new_base_values, new_data, new_display_data,
                                        feature_names=shap_values.output_names)
        shap.plots.beeswarm(shap_values_, order=class_order, show=False, plot_size=plot_size, **kwargs)
        plt.title(shap_values.feature_names[feature_idx])


def _make_plot_colors(n_values, fill=None, color=None):
    if color is None:
        color = np.repeat(False, n_values)
    if fill is None:
        fill = np.repeat(True, n_values)
    if color.dtype == bool:
        return _make_binary_plot_colors(n_values, fill, color)
    elif color.dtype == float:
        return _make_continuous_plot_colors(n_values, fill, color)
    elif color.dtype.type == np.str_:
        return _make_str_plot_colors(n_values, fill, color)
    else:
        raise Exception('color only supports a bool or number array')


def _make_binary_plot_colors(n_values, fill=None, color=None):
    draw_steps = np.repeat(0, n_values)
    draw_steps[~fill & color] = 1
    draw_steps[fill & ~color] = 2
    draw_steps[fill & color] = 3

    plot_colors = np.repeat('xkcd:light blue', n_values)
    plot_colors[draw_steps == 1] = 'xkcd:pale pink'
    plot_colors[draw_steps == 2] = 'xkcd:blue'
    plot_colors[draw_steps == 3] = 'xkcd:red'

    return draw_steps, plot_colors, None


def _make_continuous_plot_colors(n_values, fill=None, color=None):
    draw_steps = np.repeat(0, n_values)
    if fill is not None and np.sum(fill) > 0:
        draw_steps[fill] = 1
    plot_colors = color
    cmaps = ['RdYlBu_r', red_blue]
    return draw_steps, plot_colors, cmaps


def _make_str_plot_colors(n_values, fill, color):
    color_set = np.unique(color)
    if len(color_set) > 10:
        raise Exception('supports only 10 different colors at max. currently')
    draw_steps = np.array(color)
    plot_colors = np.array([color_palette[color_set.tolist().index(c)] for c in color])
    return draw_steps, plot_colors, None


def plot_feature_dependencies(shap_values, color=None, color_label=None, fill=None, alpha=None, jitter=None,
                              vlines=None, figsize=None, fig=None, axs=None):
    shap_values = ensure_shap_values_are_3d(shap_values)
    if figsize is None:
        figsize = (7, 5)
    ncols = len(shap_values.output_names)
    nrows = len(shap_values.feature_names)
    if fig is None or axs is None:
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figsize[0]*ncols, figsize[1]*nrows),
                                sharey='row', squeeze=False)
    else:
        if len(axs.shape) == 1:
            axs = axs.reshape((1, axs.shape[0]))
    sc = None
    for row_idx, (axs_row, feature) in enumerate(zip(axs, shap_values.feature_names)):
        for col_idx, (class_name, ax) in enumerate(zip(shap_values.output_names, axs_row)):
            sc = plot_feature_dependence(shap_values[:, :, class_name], feature, color=color, fill=fill, alpha=alpha,
                                          ax=ax,
                                          title=class_name if row_idx == 0 else None,
                                          jitter=jitter[row_idx] if jitter is not None else None,
                                          vlines=vlines[row_idx] if vlines is not None else None)
            if col_idx > 0:
                ax.set_ylabel(None)
                plt.setp(ax.get_yticklabels(), visible=False)
                #plt.setp(ax.get_yticklines(), visible=False)
    plt.subplots_adjust(wspace=.0)
    if color_label is not None and sc is not None:
        fig.colorbar(sc, ax=axs.ravel().tolist(), label=color_label)


def plot_feature_dependence(shap_values, feature, title=None, color=None, color_label=None, fill=None, alpha=None,
                            fig=None, ax=None, jitter=False, vlines=None):
    if isinstance(feature, (int, np.integer)):
        feature = shap_values.feature_names[feature]
    s = shap_values[:, feature]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    draw_steps, plot_colors, cmaps = _make_plot_colors(len(shap_values), fill, color)
    sc = None
    for draw_step in np.unique(draw_steps):
        cmap = None if cmaps is None else cmaps[draw_step]
        mask = draw_steps == draw_step
        s_plot = s[mask]
        current_plot_colors = plot_colors[mask]
        sc = _scatter(s_plot.data, s_plot.values, ax=ax, jitter=jitter, c=current_plot_colors, cmap=cmap, alpha=alpha, label=draw_step)
    if vlines is not None:
        for x in vlines:
            ax.axvline(x, alpha=0.6, linewidth=1, color='black', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel('SHAP value of ' + feature)
    if draw_steps.dtype.type == np.str_:
        ax.legend()
    if sc is not None and fig is not None:
        fig.colorbar(sc, ax=ax, label=color_label)
    return sc


def _scatter(x, y, ax=None, jitter=False, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    """source: https://stackoverflow.com/questions/8671808/matplotlib-avoiding-overlapping-datapoints-in-a-scatter-dot-beeswarm-plot"""
    if ax is not None:
        func = ax.scatter
    else:
        func = plt.scatter
    if jitter:
        if x.dtype == int or x.dtype == float:
            x = _rand_jitter(x)
        if y.dtype == int or y.dtype == float:
            y = _rand_jitter(y)
    return func(x, y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)


def _rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot_forces(shap_values, title=None, instance_order=None, class_order=None, **kwargs):
    """ Create force plots of all instances

    Further keyword arguments are passed to shap plot function
    e.g. link='logit'
    """
    if len(shap_values.shape) <= 2:
        return _plot_forces_singleclass(shap_values, title, instance_order, **kwargs)
    if len(shap_values.shape) == 3:
        return _plot_forces_multiclass(shap_values, instance_order, class_order, **kwargs)
    raise Exception(f'invalid dimensions: {shap_values.shape}')


def _plot_forces_singleclass(shap_values, title=None, instance_order=None, **kwargs):
    if instance_order is not None and isinstance(instance_order, np.ndarray):
        instance_order = instance_order.tolist()
    X_display = _get_display_data(shap_values)
    plot = shap.plots.force(
        base_value=shap_values.base_values[0],
        shap_values=shap_values.values,
        features=X_display,
        feature_names=shap_values.feature_names,
        out_names=title,
        ordering_keys=instance_order,
        **kwargs)
    display(plot)


def _plot_forces_multiclass(shap_values, instance_order=None, class_order=None, **kwargs):
    if class_order is None:
        class_order = range(shap_values.shape[2])
    if instance_order is not None and isinstance(instance_order, np.ndarray):
        instance_order = instance_order.tolist()
    X_display = _get_display_data(shap_values)
    for class_idx in class_order:
        shap_values_ = shap_values[:, :, class_idx]
        plot = shap.plots.force(
            base_value=shap_values_.base_values[0],
            shap_values=shap_values_.values,
            features=X_display,
            feature_names=shap_values.feature_names,
            out_names=str(shap_values_.output_names),
            ordering_keys=instance_order,
            **kwargs)
        display(plot)


def plot_decisions(shap_values, classes=None, **kwargs):
    if len(shap_values.shape) == 2:
        _plot_decision_singleclass(shap_values, **kwargs)
    elif len(shap_values.shape) == 3:
        if classes is None:
            classes = shap_values.output_names
        for class_ in classes:
            _plot_decision_singleclass(shap_values[:, :, class_], **kwargs)
    else:
        raise Exception(f'invalid dimensions: {shap_values.shape}')


def _plot_decision_singleclass(shap_values, **kwargs):
    plt.title(shap_values.output_names)
    shap.decision_plot(shap_values.base_values[0], shap_values.values, shap_values.feature_names, **kwargs)


def print_rules(rules):
    for idx, rule in enumerate(rules, 1):
        print(f'{idx}. {rule}')
