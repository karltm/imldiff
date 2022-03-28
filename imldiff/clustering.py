from pprint import pprint
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
from explainers import plot_feature_dependencies, calc_feature_order
from util import RuleClassifier, constraint_matrix_to_rules, find_counterfactuals, \
    counterfactuals_to_constraint_matrix, evaluate
from util import evaluate_predictions
from sklearn.neighbors import KNeighborsClassifier


class Explanation:
    def __init__(self, comparer, shap_values, instance_indices, cluster_classes, pred_classes, categorical_features,
                 feature_precisions, highlight, focus_class=None, parent=None, counterfactuals=None):
        self.comparer = comparer
        self.parent = parent
        self.root = self if parent is None else parent.root
        self.instance_indices = instance_indices
        self.orig_shap_values = shap_values
        self.shap_values = shap_values[instance_indices]
        self.data = pd.DataFrame(self.shap_values.data, columns=self.shap_values.feature_names, index=instance_indices)
        self.focus_class = focus_class
        self.feature_precisions = feature_precisions
        self.orig_highlight = highlight
        self.highlight = highlight[instance_indices]
        self.cluster_classes = cluster_classes
        self.categorical_features = categorical_features
        self.orig_pred_classes = pred_classes
        self.pred_classes = pred_classes[instance_indices]
        self.class_counts = pd.Series(self.pred_classes).value_counts()
        self._calculate_feature_order()
        self.counterfactuals = counterfactuals[str(self)] if counterfactuals is not None else (
            parent.counterfactuals if parent is not None and
                                      self.class_counts.get(self.focus_class) ==
                                      parent.class_counts.get(self.focus_class) else
            self._calculate_counterfactuals()
        )

    @property
    def features_ordered(self):
        return self.comparer.feature_names[self.feature_order]

    @property
    def features_with_counterfactuals(self):
        return np.array([feature for feature in self.features_ordered if len(self.counterfactuals[feature]) > 0])

    @property
    def features_without_counterfactuals(self):
        return self.comparer.feature_names[~np.isin(self.comparer.feature_names, self.features_with_counterfactuals)]

    def _calculate_feature_order(self):
        self.feature_order, self.feature_importances = calc_feature_order(self.shap_values)

    def _calculate_counterfactuals(self):
        counterfactuals = dict([(feature, []) for feature in self.comparer.feature_names])
        if np.sum(self.highlight) == 0 or self.focus_class is None:
            return counterfactuals
        counterfactuals |= find_counterfactuals(self.comparer, self.data.iloc[self.highlight].to_numpy(),
                                                     self.root.data.to_numpy(), self.feature_precisions,
                                                     list(self.comparer.class_names).index(self.focus_class))
        return counterfactuals

    def describe_feature_differences(self, feature):
        feature_idx, feature_name = self.comparer.check_feature(feature)
        feature_data = self.data.iloc[self.highlight, feature_idx]
        if len(feature_data) == 0:
            return
        lower_bound = feature_data.min()
        upper_bound = feature_data.max()
        if lower_bound == upper_bound:
            print(f'{feature_name} == {lower_bound}', end='')
        else:
            if lower_bound > self.root.data.iloc[:, feature_idx].min():
                print(f'{lower_bound} <= ', end='')
            print(feature_name, end='')
            if upper_bound < self.root.data.iloc[:, feature_idx].max():
                print(f' <= {upper_bound}', end='')
        print()

    def filter(self, by):
        if isinstance(by, str):
            instance_indices = self.data.query(by).index.to_numpy()
        else:
            instance_indices = self.instance_indices[by]
        return Explanation(self.comparer, self.orig_shap_values, instance_indices, self.cluster_classes, self.orig_pred_classes,
                           self.categorical_features, self.feature_precisions, self.orig_highlight,
                           self.focus_class, self)

    def plot_feature_dependence(self, *features, classes=None, alpha=0.5, color=None, fill=None, focus=None,
                                figsize=None, fig=None, axs=None, print_stats=False, show=True):
        node = focus if focus is not None else self
        if len(features) == 0:
            features = node.features_ordered
        scs = []
        for feature in features:
            if print_stats:
                node.describe_feature_differences(feature)
                pprint(node.counterfactuals[feature])
            scs.append(self._plot_feature_dependence(feature, classes, alpha, color, fill, focus, figsize, fig, axs))
            if show:
                plt.show()
        return scs

    def _plot_feature_dependence(self, feature, classes=None, alpha=0.5, color=None, fill=None, focus=None,
                                figsize=None, fig=None, axs=None):
        if focus is not None:
            fill = np.in1d(self.instance_indices, focus.instance_indices)
            counterfactuals = focus.counterfactuals
        else:
            counterfactuals = self.counterfactuals
        feature = self.comparer.check_feature(feature)[1]
        if classes is None:
            classes = self.cluster_classes
        color_feature_name = None
        if color is None:
            color = self.highlight
        elif isinstance(color, str) or isinstance(color, int):
            color_feature_idx, color_feature_name = self.comparer.check_feature(color)
            color = self.data.iloc[:, color_feature_idx]
        s = self.shap_values[:, :, classes]
        return plot_feature_dependencies(s[:, [feature]], color=color, color_label=color_feature_name, fill=fill,
                                         alpha=alpha, jitter=feature in self.categorical_features,
                                         vlines=[cf.value for cf in counterfactuals.get(feature) if cf], figsize=figsize,
                                         fig=fig, axs=axs)

    def plot_outcomes(self, classes=None, ax=None):
        classes = classes if classes is not None else self.cluster_classes
        y_pred = self.shap_values.base_values + self.shap_values.values.sum(1)
        class_mask = np.in1d(self.shap_values.output_names, classes)
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
        s = self.shap_values
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
        if self.parent is None or n == 0:
            return self
        return self.parent.get_parent(n - 1)

    def describe_feature(self, feature):
        feature_idx, feature_name = self.comparer.check_feature(feature)
        feature_data = self.data.iloc[:, feature_idx]
        root_feature_data = self.root.data.iloc[:, feature_idx]
        if feature_name in self.categorical_features:
            return pd.DataFrame([
                root_feature_data.value_counts(),
                feature_data.value_counts(),
                feature_data.iloc[self.highlight].value_counts()
            ], index=['global', 'local-all', 'local-diff'])
        else:
            return pd.DataFrame(index=['global', 'local-all', 'local-diff'], data=[
                root_feature_data,
                feature_data,
                feature_data.iloc[self.highlight]]).T.describe()

    def rule_from_counterfactuals(self, *include_features, order=None, latex=False):
        include_features = self.features_with_counterfactuals if len(include_features) == 0 else include_features
        order = self.feature_order if order is None else order

        constraint = self.constraint_matrix_from_counterfactuals(*include_features)
        rule = constraint_matrix_to_rules([constraint], self.comparer.feature_names, order, self.feature_precisions, latex)[0]
        instance_indices = self.instance_indices[self.highlight]
        return rule, constraint, instance_indices

    def constraint_matrix_from_counterfactuals(self, *include_features):
        if len(include_features) == 0:
            include_features = self.features_with_counterfactuals
        return counterfactuals_to_constraint_matrix(self.comparer.feature_names, self.feature_precisions,
                                                    include_features, self.counterfactuals)


class ExplanationNode(Explanation):
    def __init__(self, comparer, shap_values, cluster_node, instance_indices, cluster_classes, pred_classes,
                 distance_matrix, linkage_matrix, categorical_features, feature_precisions, highlight,
                 focus_class=None, parent=None, counterfactuals=None):
        self.cluster_node = cluster_node
        self.distance_matrix = distance_matrix
        self.linkage_matrix = linkage_matrix
        super(ExplanationNode, self).__init__(comparer, shap_values, instance_indices, cluster_classes,
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
        return ExplanationNode(self.comparer, self.orig_shap_values, cluster_node, instance_indices, self.cluster_classes,
                               self.orig_pred_classes, self.distance_matrix, self.linkage_matrix, self.categorical_features,
                               self.feature_precisions, self.orig_highlight, self.focus_class, self, counterfactuals)

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
            'highlight': self.orig_highlight,
            'counterfactuals': self._get_all_counterfactuals()
        }

    def _get_all_counterfactuals(self):
        childs = [n for n in [self.left, self.right] if n is not None]
        child_cfs = [child._get_all_counterfactuals().items() for child in childs]
        child_cfs = [item for sublist in child_cfs for item in sublist]
        child_cfs = [(str(self), self.counterfactuals)] + child_cfs
        child_cfs = dict(child_cfs)
        return child_cfs

    def __repr__(self):
        if self.parent is None:
            return 'root'
        if self.parent.cluster_node.get_left().id == self.cluster_node.id:
            name = 'L'
        else:
            name = 'R'
        parent_name = str(self.parent)
        if parent_name == 'root':
            return name
        else:
            return parent_name + name

    def get(self, name):
        if name.startswith('L'):
            return self.get_left().get(name[1:])
        elif name.startswith('R'):
            return self.get_right().get(name[1:])
        else:
            return self

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_last_child_before_focus_class_split(self):
        if not self.focus_class in list(self.class_counts.keys()):
            raise Exception('Difference class not present in cluster')
        if len(self.shap_values) == 1:
            return self
        left = self.get_left()
        right = self.get_right()
        if right is not None and not self.focus_class in list(left.class_counts.keys()):
            return right.get_last_child_before_focus_class_split()
        if right is not None and not self.focus_class in list(right.class_counts.keys()):
            return left.get_last_child_before_focus_class_split()
        return self

    def plot_dendrogram(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        hierarchy.dendrogram(self.linkage_matrix, orientation='right', ax=ax, no_labels=True)
        ax.set_title('Dendrogram')
        plt.show()


def make_clustering(comparer, shap_values, focus_class=None, cluster_classes=None, categorical_features=None,
                    feature_precisions=None, state=None):
    if cluster_classes is None:
        cluster_classes = shap_values.output_names
    if categorical_features is None:
        categorical_features = []
    if feature_precisions is None:
        feature_precisions = [0 for _ in range(len(comparer.feature_names))]
    if state is not None:
        return ExplanationNode(comparer, shap_values, cluster_classes=cluster_classes, focus_class=focus_class,
                               categorical_features=categorical_features, feature_precisions=feature_precisions, **state)
    values = shap_values[:, :, cluster_classes].values
    values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    distance_matrix = distance.pdist(values, metric='sqeuclidean')
    linkage_matrix = hierarchy.complete(distance_matrix)
    cluster_node = hierarchy.to_tree(linkage_matrix)
    instance_indices = np.array(cluster_node.pre_order())
    pred_classes = comparer.predict_mclass_diff(shap_values.data)
    pred_classes = comparer.class_names[pred_classes]
    if focus_class is not None:
        highlight = pred_classes == focus_class
    else:
        highlight = comparer.predict_bin_diff(shap_values.data)
    node = ExplanationNode(comparer, shap_values, cluster_node, instance_indices, cluster_classes, pred_classes,
                           distance_matrix, linkage_matrix, categorical_features, feature_precisions, highlight,
                           focus_class)
    return node


get_node_path = lambda node: get_node_path(node.parent) + [node] if node is not None else []


def plot_joint_feature_dependence(feature, classes=None, figsize=(4, 2), with_context=False, **nodes):
    classes = next(iter(nodes.values())).cluster_classes if classes is None else classes
    context_nodes = nodes.values()
    focus_nodes = [
        node.get_last_child_before_focus_class_split() if hasattr(node, 'get_last_child_before_focus_class_split') else node
        for node in nodes.values()
    ] if with_context else nodes.values()

    ncols = len(classes)
    nrows = len(nodes)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * figsize[0], nrows * figsize[1]), sharex='all', sharey='all', squeeze=False)
    for node_name, context_node, focus_node, axs_row in zip(nodes.keys(), context_nodes, focus_nodes, axs):
        context_node.plot_feature_dependence(feature, classes=classes, fig=fig, axs=axs_row, focus=focus_node, show=False)
        axs_row[0].set_ylabel(f'{node_name}\ns({feature})')
    for axs_row in axs[1:]:
        for ax in axs_row:
            ax.set_title('')
    plt.subplots_adjust(wspace=.0, hspace=.0)


def compare_indiv_dep_plots(node: ExplanationNode, feature=None, alpha=0.5, fig=None, axs=None):
    if feature is None:
        for feature in node.features_ordered:
            _compare_indiv_dep_plots(node, feature, alpha)
            plt.show()
    else:
        _compare_indiv_dep_plots(node, feature, alpha, fig, axs)


def _compare_indiv_dep_plots(node: ExplanationNode, feature, alpha=0.5, fig=None, axs=None):
    class_names = [class_name for class_name in node.comparer.base_class_names
                   if any([n.endswith('.' + class_name) for n in node.shap_values.output_names])]
    class_names_a, class_names_b = tuple([
        [clf + '.' + class_name for class_name in class_names] for clf in ['A', 'B']
    ])
    if fig is None or axs is None:
        fig, axs = plt.subplots(ncols=len(class_names_a), nrows=2, sharex='all', sharey='row', squeeze=False,
                                figsize=(len(class_names_a) * 7, 1.5 * 5), gridspec_kw={'height_ratios': [2,1]})
    for ax in axs[0]:
        ax.axhline(0, linewidth=1, color='grey', alpha=0.5)
    scs_a = node.plot_feature_dependence(feature, classes=class_names_a, alpha=alpha*2/3,
                                         color=np.repeat(False, len(node.data)),
                                         fig=fig, axs=axs[0], show=False)
    scs_b = node.plot_feature_dependence(feature, classes=class_names_b, alpha=alpha*2/3,
                                         color=np.repeat(True, len(node.data)),
                                         fig=fig, axs=axs[0], show=False)
    for ax, label in zip(axs[0], class_names):
        ax.set_title('Class ' + label)
    axs[0][-1].legend([scs_a[-1], scs_b[-1]], ['A', 'B'])
    axs[1][0].set_ylabel('Difference')
    for class_name_a, class_name_b, ax in zip(class_names_a, class_names_b, axs[1]):
        diff = node.shap_values[:, feature, class_name_b].values - node.shap_values[:, feature, class_name_a].values
        ax.scatter(node.data[feature], diff, alpha=alpha)
        ax.axhline(0, linewidth=1, alpha=0.5)
        ax.set_xlabel(feature)
    plt.subplots_adjust(wspace=.0, hspace=.0)


def plot_2d(node: ExplanationNode, x, y):
    node.comparer.plot_decision_boundaries(node.root.data, type='bin-diffclf', x=x, y=y, alpha=0.5)
    for cf in node.counterfactuals[x]:
        plt.axvline(cf.value, linewidth=1, color='black', linestyle='--')
    for cf in node.counterfactuals[y]:
        plt.axhline(cf.value, linewidth=1, color='black', linestyle='--')


def eval_clusterings(explanations_per_class: dict[str, ExplanationNode], X_test, y_test, shap_values_test):
    metrics = []
    for class_name, explanation in explanations_per_class.items():
        if explanation.highlight.sum() == 0:
            continue
        for distance, nodes in get_nodes_per_level(explanation).items():
            metric = eval_clusterings_for_class(class_name, nodes, X_test, y_test, shap_values_test)
            metric['Distance'] = distance
            metrics.append(metric)
    return pd.DataFrame(metrics).reset_index(drop=True)


def eval_clusterings_for_class(class_name, nodes, X_test, y_test, shap_values_test):
    discr = _make_cluster_discriminator(nodes)
    pred_cluster_names = discr.predict(shap_values_test.values.reshape((shap_values_test.shape[0], -1)))
    comparer = next(iter(nodes)).comparer
    nodes = dict([(str(node), node) for node in nodes])
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
    X = np.concatenate([node.shap_values.values.reshape((node.shap_values.shape[0], -1)) for node in nodes])
    y = np.concatenate([np.repeat(str(node), len(node.shap_values.values)) for node in nodes])
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    return knn


def _filter_nodes_with_equal_distance(nodes_per_level):
    return reversed(dict([(len(nodes), (distance, nodes)) for distance, nodes in reversed(nodes_per_level)]).values())


def _nodes_flat(node: ExplanationNode):
    children = [_nodes_flat(n) for n in [node.left, node.right] if n is not None and n.highlight.sum() > 0]
    return [node] + [item for sublist in children for item in sublist]


def _cut_nodes(node: ExplanationNode, min_distance):
    if any([_has_focus_class_instances(n) and n.distance >= min_distance for n in [node.left, node.right]]):
        children = [_cut_nodes(n, min_distance) for n in [node.left, node.right] if _has_focus_class_instances(n)]
        return [item for sublist in children for item in sublist]
    else:
        return [node]


def _has_focus_class_instances(n: ExplanationNode):
    return n is not None and n.highlight.sum() > 0


def plot_2d_with_boundaries(node: ExplanationNode, x=0, y=1, fig=None, ax=None):
    X = node.root.data
    comparer = node.comparer
    x, y = comparer.check_feature(x)[1], comparer.check_feature(y)[1]
    xlim = X[x].min() - 0.5, X[x].max() + 0.5
    ylim = X[y].min() - 0.5, X[y].max() + 0.5
    comparer.plot_decision_boundaries(node.data, xlim=xlim, ylim=ylim, fig=fig, ax=ax)
    handle = plt if ax is None else ax
    for cf in node.counterfactuals['x1']:
        handle.axvline(cf.value, linewidth=1, color='black', linestyle='--')
    for cf in node.counterfactuals['x2']:
        handle.axhline(cf.value, linewidth=1, color='black', linestyle='--')
