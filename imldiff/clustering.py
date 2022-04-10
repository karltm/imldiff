import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
from explainers import calc_feature_order, make_diff_shap_values
from util import RuleClassifier, constraint_matrix_to_rules, find_counterfactuals, counterfactuals_to_constraint_matrix
from util import evaluate_predictions
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

_DEFAULT_FIGSIZE = 3.5, 2


class NodeDoesntExistException(Exception):
    pass


class Explanation:
    def __init__(self, comparer, indiv_shap_values, diffclf_shap_values, instance_indices, cluster_classes, pred_classes, categorical_features,
                 feature_precisions, highlight=None, focus_class=None, parent=None, counterfactuals=None):
        self.comparer = comparer
        self.parent = parent
        self.root = self if parent is None else parent.root
        self.instance_indices = instance_indices
        self.orig_indiv_shap_values = indiv_shap_values
        self.indiv_shap_values = indiv_shap_values[instance_indices]
        self.diff_shap_values = make_diff_shap_values(self.indiv_shap_values)
        self.orig_diffclf_shap_values = diffclf_shap_values
        self.diffclf_shap_values = diffclf_shap_values[instance_indices]
        self.data = pd.DataFrame(self.diffclf_shap_values.data, columns=self.diffclf_shap_values.feature_names, index=instance_indices)
        self.focus_class = focus_class
        self.feature_precisions = feature_precisions
        self.cluster_classes = cluster_classes
        self.categorical_features = categorical_features
        self.orig_pred_classes = pred_classes
        self.pred_classes = pred_classes[instance_indices]
        self.occuring_classes = np.unique(pred_classes) if parent is None else parent.occuring_classes
        self.class_counts = pd.Series(self.pred_classes).value_counts()
        self.diffclf_feature_order, self.diffclf_feature_importances = calc_feature_order(self.diffclf_shap_values)
        self.diff_feature_order, self.diff_feature_importances = calc_feature_order(self.diff_shap_values)
        self.counterfactuals = self._get_or_create_counterfactuals(counterfactuals)

    def _get_or_create_counterfactuals(self, counterfactuals):
        if counterfactuals is not None:
            return counterfactuals[repr(self)]
        if self.parent is not None and\
            self.class_counts.get(self.focus_class) ==\
            self.parent.class_counts.get(self.focus_class):
            return self.parent.counterfactuals
        return self._calculate_counterfactuals()

    def _calculate_counterfactuals(self):
        counterfactuals = dict([(feature, []) for feature in self.comparer.feature_names])
        if np.sum(self.highlight) == 0 or self.focus_class is None:
            return counterfactuals
        counterfactuals |= find_counterfactuals(self.comparer, self.data.iloc[self.highlight].to_numpy(),
                                                self.root.data.to_numpy(), self.feature_precisions,
                                                list(self.comparer.class_names).index(self.focus_class))
        return counterfactuals

    @property
    def highlight(self):
        if self.focus_class is None:
            return np.in1d(self.pred_classes, self.comparer.difference_class_names)
        return self.pred_classes == self.focus_class

    @property
    def features_ordered(self):
        return self.comparer.feature_names[self.diffclf_feature_order]

    @property
    def features_with_counterfactuals(self):
        return np.array([feature for feature in self.features_ordered if len(self.get_counterfactuals()[feature]) > 0])

    def get_counterfactuals(self):
        if self.focus_class is None or self.focus_class not in self.counterfactuals:
            return dict([(feature, []) for feature in self.comparer.feature_names])
        return self.counterfactuals[self.focus_class]

    @property
    def features_without_counterfactuals(self):
        return self.comparer.feature_names[~np.isin(self.comparer.feature_names, self.features_with_counterfactuals)]

    def filter(self, by):
        if isinstance(by, str):
            instance_indices = self.data.query(by).index.to_numpy()
        else:
            instance_indices = self.instance_indices[by]
        return Explanation(self.comparer, self.orig_indiv_shap_values, self.orig_diffclf_shap_values, instance_indices,
                           self.cluster_classes, self.orig_pred_classes, self.categorical_features,
                           self.feature_precisions, None, self.focus_class, self)

    def plot_indiv_feature_dependence(self, *features, classes=None, color=None, color_label=None, figsize=_DEFAULT_FIGSIZE,
                                      alpha=None, axs=None, simplify=False, show_legend=True, separate_rows=False):
        if features is None or len(features) == 0:
            features = self.comparer.feature_names[self.diff_feature_order]
        for feature in features:
            plot_indiv_dependence_curve_comparison_for_feature(self, feature, classes, simplify=simplify, color=color,
                                                               color_label=color_label, alpha=alpha, figsize=figsize,
                                                               axs=axs, show_legend=show_legend, separate_rows=separate_rows)
            if len(features) > 1:
                plt.show()

    def plot_feature_dependence(self, *features, classes=None, focus=None, color=None, color_label=None, figsize=_DEFAULT_FIGSIZE,
                                alpha=None, axs=None, simplify=False, show_legend=True):
        node = self if focus is None else focus
        if focus is None:
            plot_dependence_curves(self, features, classes, simplify=simplify, color=color,
                                   color_label=color_label, alpha=alpha, axs=axs, figsize=figsize, show_legend=show_legend)
        else:
            features = node.features_ordered if len(features) == 0 else features
            for feature in features:
                plot_dependence_curves_for_nodes(self, focus, feature=feature, labels=classes, simplify=simplify,
                                                 color=color, color_label=color_label, alpha=alpha, axs=axs,
                                                 figsize=figsize, show_legend=show_legend)
                if len(features) > 1:
                    plt.show()

    def plot_outcomes(self, classes=None, ax=None):
        classes = classes if classes is not None else self.cluster_classes
        y_pred = self.diffclf_shap_values.base_values + self.diffclf_shap_values.values.sum(1)
        class_mask = np.in1d(self.diffclf_shap_values.output_names, classes)
        y_pred = y_pred[:, class_mask]
        index = pd.Index(self.pred_classes, name='Label')
        y_pred = pd.DataFrame(y_pred, columns=classes, index=index).reset_index()
        df = pd.melt(y_pred, value_vars=classes, id_vars='Label')
        legend_labels = self.comparer.class_names[np.in1d(self.comparer.class_names, np.unique(self.pred_classes))]
        sns.stripplot(data=df, x='variable', y='value', hue='Label', hue_order=legend_labels, dodge=True, ax=ax)

    def plot_outcome_differences(self, classes=None):
        focus_class_idx = list(self.comparer.class_names).index(self.focus_class)
        other_classes = classes if classes is not None else [class_ for class_ in self.cluster_classes if class_ != self.focus_class]
        other_class_indices = [list(self.comparer.class_names).index(class_) for class_ in other_classes]
        s = self.diffclf_shap_values
        log_odds_per_class = s.base_values + s.values.sum(1)
        log_odds_of_focus_class = log_odds_per_class[:, focus_class_idx]
        log_odds_of_other_classes = log_odds_per_class[:, other_class_indices]
        log_odds_diff = log_odds_of_other_classes - np.array([log_odds_of_focus_class for _ in other_class_indices]).T

        classes = np.concatenate([np.repeat(class_, len(log_odds_diff)) for class_ in other_classes])
        class_labels = np.concatenate([self.pred_classes for _ in other_class_indices])
        df = pd.DataFrame([classes, log_odds_diff.T.flatten(), class_labels],
                          index=['class', 'difference', 'actual class']).T
        chart = sns.catplot(x='class', y='difference', hue='actual class', data=df, alpha=0.6, aspect=2, legend=False)
        chart.axes[0][0].axhline(y=0, color='black', linewidth=1, alpha=0.5)
        plt.legend(loc='upper right')
        diff_ranges_of_focus_class = [f'{round(col.min(), 2)} to {round(col.max(), 2)}'
                                     for col in log_odds_diff[self.highlight].T]
        print(dict(zip(other_classes, diff_ranges_of_focus_class)))

    def get_parent(self, n=1):
        if n == 0:
            return self
        if self.parent is None:
            raise NodeDoesntExistException()
        return self.parent.get_parent(n - 1)

    def rule_from_counterfactuals(self, *include_features, order=None, latex=False):
        include_features = self.features_with_counterfactuals if len(include_features) == 0 else include_features
        order = self.diffclf_feature_order if order is None else order

        constraint = self.constraint_matrix_from_counterfactuals(*include_features)
        rule = constraint_matrix_to_rules([constraint], self.comparer.feature_names, order, self.feature_precisions, latex)[0]
        instance_indices = self.instance_indices[self.highlight]
        return rule, constraint, instance_indices

    def constraint_matrix_from_counterfactuals(self, *include_features):
        if len(include_features) == 0:
            include_features = self.features_with_counterfactuals
        return counterfactuals_to_constraint_matrix(self.comparer.feature_names, self.feature_precisions,
                                                    include_features, self.get_counterfactuals())


class ExplanationNode(Explanation):
    def __init__(self, comparer, indiv_shap_values, diffclf_shap_values, cluster_node, instance_indices, cluster_classes, pred_classes,
                 distance_matrix, linkage_matrix, categorical_features, feature_precisions, highlight,
                 focus_class=None, parent=None, counterfactuals=None):
        self.cluster_node = cluster_node
        self.distance_matrix = distance_matrix
        self.linkage_matrix = linkage_matrix
        self.cluster_name = _make_cluster_name(parent, cluster_node)
        super(ExplanationNode, self).__init__(comparer, indiv_shap_values, diffclf_shap_values, instance_indices, cluster_classes,
                                              pred_classes, categorical_features, feature_precisions,
                                              highlight, focus_class, parent, counterfactuals)
        self.left = self._make_child(self.cluster_node.get_left(), counterfactuals)
        self.right = self._make_child(self.cluster_node.get_right(), counterfactuals)

    def _make_child(self, cluster_node, counterfactuals=None):
        if self.distance == 0.0:
            return None
        if cluster_node is None:
            return None
        instance_indices = np.array(cluster_node.pre_order())
        return ExplanationNode(self.comparer, self.orig_indiv_shap_values, self.orig_diffclf_shap_values, cluster_node,
                               instance_indices, self.cluster_classes, self.orig_pred_classes, self.distance_matrix,
                               self.linkage_matrix, self.categorical_features, self.feature_precisions,
                               None, self.focus_class, self, counterfactuals)

    def __repr__(self):
        return self.cluster_name

    @property
    def distance(self):
        """Distance for this cluster in the distance matrix (metric='sqeuclidean')"""
        return self.cluster_node.dist

    @property
    def state(self):
        return {
            'cluster_node': self.cluster_node,
            'instance_indices': self.instance_indices,
            'pred_classes': self.orig_pred_classes,
            'distance_matrix': self.distance_matrix,
            'linkage_matrix': self.linkage_matrix,
            'counterfactuals': self._get_all_counterfactuals()
        }

    def _get_all_counterfactuals(self):
        childs = [n for n in [self.left, self.right] if n is not None]
        child_cfs = [child._get_all_counterfactuals().items() for child in childs]
        child_cfs = [item for sublist in child_cfs for item in sublist]
        child_cfs = [(repr(self), self.counterfactuals)] + child_cfs
        child_cfs = dict(child_cfs)
        return child_cfs

    def get(self, name):
        if name.startswith('L'):
            return self.get_left().get(name[1:])
        elif name.startswith('R'):
            return self.get_right().get(name[1:])
        else:
            return self

    def get_left(self):
        node = self.left
        if node is not None:
            node.focus_class = self.focus_class
        return node

    def get_right(self):
        node = self.right
        if node is not None:
            node.focus_class = self.focus_class
        return node

    def descend(self):
        """Descend until the focus instances would be split into separate clusters"""
        if len(self.data) == 1:
            return self
        left = self.get_left()
        right = self.get_right()
        if left is not None and np.array_equal(self.instance_indices[self.highlight], left.instance_indices[left.highlight]):
            return left.descend()
        if right is not None and np.array_equal(self.instance_indices[self.highlight], right.instance_indices[right.highlight]):
            return right.descend()
        return self

    def plot_dendrogram(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        hierarchy.dendrogram(self.linkage_matrix, orientation='right', ax=ax, no_labels=True)
        ax.set_title('Dendrogram')
        plt.show()


def _make_cluster_name(parent, cluster_node):
    if parent is None:
        return 'root'
    is_left_child = parent.cluster_node.get_left().id == cluster_node.id
    if is_left_child:
        name = 'L'
    else:
        name = 'R'
    parent_name = parent.cluster_name
    if parent_name == 'root':
        return name
    else:
        return parent_name + name


def make_clustering(comparer, indiv_shap_values, diffclf_shap_values, focus_class=None, cluster_classes=None, categorical_features=None,
                    feature_precisions=None, state=None):
    if cluster_classes is None:
        cluster_classes = diffclf_shap_values.output_names
    if categorical_features is None:
        categorical_features = []
    if feature_precisions is None:
        feature_precisions = [0 for _ in range(len(comparer.feature_names))]
    if state is not None:
        return ExplanationNode(comparer, indiv_shap_values, diffclf_shap_values, cluster_classes=cluster_classes, focus_class=focus_class,
                               categorical_features=categorical_features, feature_precisions=feature_precisions, **state)
    values = diffclf_shap_values[:, :, cluster_classes].values
    values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    distance_matrix = distance.pdist(values, metric='sqeuclidean')
    linkage_matrix = hierarchy.complete(distance_matrix)
    cluster_node = hierarchy.to_tree(linkage_matrix)
    instance_indices = np.array(cluster_node.pre_order())
    X = diffclf_shap_values.data
    pred_classes = comparer.predict_mclass_diff(X)
    pred_classes = comparer.class_names[pred_classes]
    if focus_class is not None:
        highlight = pred_classes == focus_class
    else:
        highlight = comparer.predict_bin_diff(X)
    node = ExplanationNode(comparer, indiv_shap_values, diffclf_shap_values, cluster_node, instance_indices, cluster_classes, pred_classes,
                           distance_matrix, linkage_matrix, categorical_features, feature_precisions, highlight,
                           focus_class)
    return node


def eval_clusterings(node: ExplanationNode, X_test, shap_values_test):
    node = node.root
    metrics = []
    for class_name in node.comparer.class_names:
        node.focus_class = class_name
        if node.highlight.sum() == 0:
            continue
        for distance, nodes in get_nodes_per_level(node).items():
            metric = eval_clusterings_for_class(class_name, nodes, X_test, shap_values_test)
            metric['Distance'] = distance
            metrics.append(metric)
    return pd.DataFrame(metrics).reset_index(drop=True)


def eval_clusterings_for_class(class_name, nodes, X_test, shap_values_test):
    discr = _make_cluster_discriminator(nodes)
    pred_cluster_names = discr.predict(shap_values_test.values.reshape((shap_values_test.shape[0], -1)))
    nodes = dict([(str(node), node) for node in nodes])
    comparer = next(iter(nodes.values())).comparer
    y_test = comparer.predict_mclass_diff(X_test)
    y_pred = np.repeat(False, len(y_test))
    constraints = []
    for cluster_name in np.unique(pred_cluster_names):
        mask = pred_cluster_names == cluster_name
        node = nodes[cluster_name]
        rule, constraint, _ = node.rule_from_counterfactuals()
        constraints.append(constraint)
        rclf = RuleClassifier(comparer.feature_names, [rule])
        y_pred[mask] = rclf.predict(X_test[mask])
    y_true = comparer.class_names[y_test] == class_name
    metric = evaluate_predictions(y_true, y_pred, [False, True], [False, True]).iloc[1, :].copy()
    metric['Label'] = class_name
    metric['Nodes'] = len(nodes)
    metric['Constraints'] = np.sum(~np.isnan(constraints))
    return metric


def get_nodes_per_level(node):
    distances = reversed(np.unique([node.distance for node in _nodes_flat(node)]))
    nodes_per_level = [(distance, _cut_nodes(node, distance)) for distance in distances]
    nodes_per_level = _filter_nodes_with_equal_distance(nodes_per_level)
    return dict(nodes_per_level)


def _make_cluster_discriminator(nodes):
    X = np.concatenate([node.diffclf_shap_values.values.reshape((node.diffclf_shap_values.shape[0], -1)) for node in nodes])
    y = np.concatenate([np.repeat(str(node), len(node.diffclf_shap_values.values)) for node in nodes])
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    return knn


def _filter_nodes_with_equal_distance(nodes_per_level):
    return reversed(dict([(len(nodes), (distance, nodes)) for distance, nodes in reversed(nodes_per_level)]).values())


def _nodes_flat(node: ExplanationNode):
    children = [_nodes_flat(n) for n in [node.descend().get_left(), node.descend().get_right()] if n is not None and n.highlight.sum() > 0]
    return [node] + [item for sublist in children for item in sublist]


def _cut_nodes(node: ExplanationNode, min_distance):
    if any([_has_focus_class_instances(n) and n.distance >= min_distance for n in [node.descend().get_left(), node.descend().get_right()]]):
        children = [_cut_nodes(n, min_distance) for n in [node.descend().get_left(), node.descend().get_right()] if _has_focus_class_instances(n)]
        return [item for sublist in children for item in sublist]
    else:
        return [node]


def _has_focus_class_instances(n: ExplanationNode):
    return n is not None and n.highlight.sum() > 0


def plot_2d(node: ExplanationNode, x, y):
    node.comparer.plot_decision_boundaries(node.root.data, type='bin-diffclf', x=x, y=y, alpha=0.5)
    for cf in node.get_counterfactuals()[x]:
        plt.axvline(cf.value, linewidth=1, color='black', linestyle='--')
    for cf in node.get_counterfactuals()[y]:
        plt.axhline(cf.value, linewidth=1, color='black', linestyle='--')


def plot_2d_with_boundaries(node: ExplanationNode, x=0, y=1, fig=None, ax=None):
    X = node.root.data
    comparer = node.comparer
    x, y = comparer.check_feature(x)[1], comparer.check_feature(y)[1]
    xlim = X[x].min() - 0.5, X[x].max() + 0.5
    ylim = X[y].min() - 0.5, X[y].max() + 0.5
    comparer.plot_decision_boundaries(node.data, xlim=xlim, ylim=ylim, fig=fig, ax=ax)
    handle = plt if ax is None else ax
    for cf in node.get_counterfactuals()[x]:
        handle.axvline(cf.value, linewidth=1, color='black', linestyle='--')
    for cf in node.get_counterfactuals()[y]:
        handle.axhline(cf.value, linewidth=1, color='black', linestyle='--')


def plot_dependence_curves_for_nodes(*nodes, feature, labels=None, kind='diffclf', simplify=False, color=None, color_label=None, alpha=None, axs=None, figsize=_DEFAULT_FIGSIZE, show_legend=True, separate_rows=False, adjust=True, **kw_nodes):
    kw_nodes |= dict([(str(node), node) for node in nodes])
    first_node = next(iter(kw_nodes.values()))
    labels = _get_labels(first_node, kind, labels)
    nrows, ncols = len(kw_nodes), len(labels)
    axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*figsize[0], nrows*figsize[1]),
                       sharey='row' if separate_rows else 'all', sharex='row' if separate_rows else 'all',
                       squeeze=False, constrained_layout=True)[1] if axs is None else axs
    have_equal_focus_instances = _nodes_have_equal_focus_instances(kw_nodes.values())
    for idx, (node_name, node, axs_row) in enumerate(zip(kw_nodes.keys(), kw_nodes.values(), axs)):
        is_first, is_last = idx == 0, idx == len(kw_nodes) - 1
        plot_dependence_curves_for_feature(node, feature, labels=labels, kind=kind, simplify=simplify, color=color,
                                           color_label=color_label, alpha=alpha, show_title=is_first,
                                           show_cf_legend=not have_equal_focus_instances or is_last,
                                           show_label_legend=is_first and show_legend, axs=axs_row, adjust=False)
        axs_row[0].set_ylabel(node_name + '\n' + axs_row[0].get_ylabel())
    if adjust:
        if separate_rows:
            plt.subplots_adjust(wspace=.02)
        else:
            plt.subplots_adjust(wspace=.02, hspace=.02)
        draw_colorbar_if_necessary(first_node, color, color_label)


def _nodes_have_equal_focus_instances(nodes):
    indices_arr = [node.instance_indices[node.highlight] for node in nodes]
    first = indices_arr[0]
    return all([np.array_equal(indices, first) for indices in indices_arr])


def _get_labels(node, kind, labels=None):
    if labels is not None:
        return labels
    if kind == 'indiv':
        return node.indiv_shap_values.output_names
    if kind == 'indiv-diff':
        return node.diff_shap_values.output_names
    else:
        return node.cluster_classes


def plot_dependence_curves(node, features=None, labels=None, kind='diffclf', simplify=False, color=None, color_label=None, alpha=None, axs=None, figsize=_DEFAULT_FIGSIZE, show_legend=True, adjust=True):
    features = node.features_ordered if features is None else features
    labels = _get_labels(node, kind, labels)
    nrows, ncols = len(features), len(labels)
    axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*figsize[0], nrows*figsize[1]), sharey='row', squeeze=False, constrained_layout=True)[1] if axs is None else axs
    for idx, (feature, axs_row) in enumerate(zip(features, axs)):
        is_first = idx == 0
        plot_dependence_curves_for_feature(node, feature, labels=labels, kind=kind, simplify=simplify, color=color, color_label=color_label, alpha=alpha, show_title=is_first, axs=axs_row, show_label_legend=show_legend, adjust=False)
    if adjust:
        plt.subplots_adjust(wspace=.02, hspace=.02)
        draw_colorbar_if_necessary(node, color, color_label)


def plot_dependence_curves_for_feature(node, feature, labels=None, kind='diffclf', simplify=False, color=None, color_label=None, alpha=None, axs=None, show_title=True, show_cf_legend=True, show_label_legend=True, figsize=_DEFAULT_FIGSIZE, adjust=True):
    labels = _get_labels(node, kind, labels)
    ncols = len(labels)
    axs = plt.subplots(ncols=ncols, figsize=(ncols*figsize[0], figsize[1]), sharey='row', squeeze=False, constrained_layout=True)[1][0] if axs is None else axs
    for idx, (label, ax) in enumerate(zip(labels, axs)):
        is_first, is_last = idx == 0, idx == len(labels) - 1
        ax = plot_dependence_curve(node, feature, label, kind=kind, simplify=simplify, color=color, color_label=color_label, alpha=alpha, ax=ax, show_cf_legend=is_last and show_cf_legend, show_label_legend=is_last and show_label_legend)
        if show_title:
            ax.set_title(f'Class {label}\'s outcome')
        if not is_first:
            ax.set_ylabel('')
    if kind == 'indiv-diff':
        axs[0].set_ylabel(f'$s_{node.comparer.name_b}-s_{node.comparer.name_a}$')
    if adjust:
        plt.subplots_adjust(wspace=.02, hspace=.02)
        draw_colorbar_if_necessary(node, color, color_label)


def plot_dependence_curve(node, feature, label, kind='diffclf', simplify=False, color=None, color_label=None, alpha=None, ax=None, show_cf_legend=True, show_label_legend=True):
    class_names = node.occuring_classes
    if kind == 'indiv':
        s = node.indiv_shap_values[:, feature, label]
    elif kind == 'indiv-diff':
        s = node.diff_shap_values[:, feature, label]
    else:
        s = node.diffclf_shap_values[:, feature, label]
    df = pd.DataFrame({
        feature: node.data[feature],
        f's({feature})': s.values,
        'Label': node.pred_classes,
    })
    df = pd.concat([df.loc[~node.highlight, :], df.loc[node.highlight]])
    if simplify:
        if node.focus_class is not None:
            class_names = ['other', node.focus_class]
            df.loc[df['Label'] != node.focus_class, 'Label'] = 'other'
        else:
            class_names = ['equal', 'different'] if node.focus_class is None else ['other', node.focus_class]
            df['Label'] = [class_names[1] if is_highlighted else class_names[0] for is_highlighted in node.highlight]
    if color is not None:
        if color_label is not None:
            df[color_label] = color
        else:
            color_label = color
            df[color_label] = node.data[color]
        hue_order = None
        palette = 'tab10' if _is_categorical_color(node, color) else 'flare_r'
    else:
        color_label = 'Label'
        hue_order = class_names
        palette = _get_colors(node)
    if feature in node.categorical_features:
        df[feature] = _jitter(df[feature])
    ax = sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue=color_label, hue_order=hue_order, palette=palette, alpha=alpha, ax=ax, linewidth=0)
    if kind == 'indiv-diff':
        ax.axhline(0, linewidth=1, alpha=0.33, color='black')
    if show_label_legend and _is_categorical_color(node, color):
        legend_sc = ax.legend(title=color_label, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        ax.add_artist(legend_sc)
    else:
        ax.get_legend().remove()
    lines = [_plot_counterfactual(cf, ax=ax) for cf in node.get_counterfactuals()[feature]]
    if show_cf_legend and len(lines) > 0:
        ax.legend(lines, [line.get_label() for line in lines], title='Counterfactuals')
    if node.feature_precisions[list(node.comparer.feature_names).index(feature)] == 0:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
    return ax


def _jitter(values, j=0):
    return values + np.random.normal(j, 0.1, values.shape)


def _is_categorical_color(node, color):
    if color is None:
        return True
    if isinstance(color, str):
        color = node.data[color]
    return np.issubdtype(color.dtype, str)


def _get_colors(node):
    class_mask = np.in1d(node.occuring_classes, node.comparer.difference_class_names)
    get_bright_color = lambda idx: sns.color_palette('bright')[idx]
    get_deep_color = lambda idx: sns.color_palette('deep')[idx]
    indices = np.arange(len(node.occuring_classes))
    indices = np.argsort(list(indices[class_mask]) + list(indices[~class_mask]))
    colors = [get_bright_color(idx) if is_bright else get_deep_color(idx) for idx, is_bright in zip(indices, class_mask)]
    return colors


def _plot_counterfactual(cf, ax):
    linestyle = 'dotted' if cf.is_direction_up else 'dashed'
    return ax.axvline(cf.value, alpha=0.3, linewidth=2, color='black', linestyle=linestyle, label=str(cf))


def plot_indiv_dependence_curve_comparison_for_feature(node, feature, labels=None, simplify=False, color=None, color_label=None, alpha=None, figsize=_DEFAULT_FIGSIZE, axs=None, show_legend=True, adjust=True, separate_rows=False):
    comparer = node.comparer
    base_labels = comparer.base_class_names if labels is None else labels
    nrows, ncols = 3, len(base_labels)
    axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*figsize[0], nrows*figsize[1]), sharex='all', sharey='row' if separate_rows else 'all', squeeze=False, constrained_layout=True)[1] if axs is None else axs
    get_labels = lambda clf: [clf + '.' + label for label in base_labels]
    plot_dependence_curves_for_feature(node, feature, get_labels(comparer.name_a), kind='indiv', simplify=simplify, color=color, color_label=color_label, alpha=alpha, axs=axs[0], show_title=False, show_label_legend=False, adjust=False)
    plot_dependence_curves_for_feature(node, feature, get_labels(comparer.name_b), kind='indiv', simplify=simplify, color=color, color_label=color_label, alpha=alpha, axs=axs[1], show_title=False, show_label_legend=show_legend, adjust=False)
    plot_dependence_curves_for_feature(node, feature, base_labels, kind='indiv-diff', color=color, color_label=color_label, alpha=alpha, axs=axs[2], show_label_legend=False, adjust=False)
    for i in range(ncols):
        axs[0][i].set_title(axs[2][i].get_title())
        axs[2][i].set_title(None)
    axs[0][0].set_ylabel(f'$s_{comparer.name_a}$({feature})')
    axs[1][0].set_ylabel(f'$s_{comparer.name_b}$({feature})')
    if adjust:
        plt.subplots_adjust(wspace=.02, hspace=.02)
        draw_colorbar_if_necessary(node, color, color_label)


def draw_colorbar_if_necessary(node, color, color_label=None, fig=None, axs=None):
    if color is None or _is_categorical_color(node, color):
        return
    if color_label is None:
        color_label = color
        color = node.data[color]
    norm = plt.Normalize(color.min(), color.max())
    sm = plt.cm.ScalarMappable(cmap='flare_r', norm=norm)
    sm.set_array([])
    fig = plt.gca().figure if fig is None else fig
    axs = fig.axes if axs is None else axs
    fig.colorbar(sm, ax=axs, label=color_label, shrink=0.9, pad=0.01)
