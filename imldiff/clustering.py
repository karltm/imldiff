import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from explainers import plot_feature_dependencies_for_classes, calc_feature_order
from util import index_of


class Counterfactual:
    def __init__(self, feature, value, is_direction_up, outcomes):
        self.feature = feature
        self.value = value
        self.is_direction_up = is_direction_up
        self.outcomes = outcomes

    def is_including_class(self, class_):
        return class_ in self.outcomes.keys()

    def __repr__(self):
        sign = '>=' if self.is_direction_up else '<='
        return f'{self.feature} {sign} {self.value} --> {self.outcomes}'


class Explanation:
    def __init__(self, comparer, orig_shap_values, instance_indices, cluster_classes, orig_pred_classes,
                 categorical_features, feature_precisions, orig_highlight, diff_class=None, parent=None):
        self.comparer = comparer
        self.parent = parent
        self.root = self if parent is None else parent.root
        self.instance_indices = instance_indices
        self.orig_shap_values = orig_shap_values
        self.shap_values = orig_shap_values[instance_indices]
        self.data = self.shap_values.data
        self.diff_class = diff_class
        self.feature_precisions = feature_precisions
        self.orig_highlight = orig_highlight
        self.highlight = orig_highlight[instance_indices]
        self.cluster_classes = cluster_classes
        self.categorical_features = categorical_features
        self.orig_pred_classes = orig_pred_classes
        self.pred_classes = orig_pred_classes[instance_indices]
        self.class_counts = pd.Series(self.pred_classes).value_counts()
        self._calculate_feature_order()
        if parent is not None and self.class_counts.get(self.diff_class) == parent.class_counts.get(self.diff_class):
            self.counterfactuals = parent.counterfactuals
        else:
            self._calculate_counterfactuals()

    @property
    def features_ordered(self):
        return self.comparer.feature_names[self.feature_order]

    @property
    def features_with_counterfactuals(self):
        mask = np.isin(self.features_ordered, list(self.counterfactuals.keys()))
        return self.features_ordered[mask]

    @property
    def features_without_counterfactuals(self):
        return self.comparer.feature_names[~np.isin(self.comparer.feature_names, self.features_with_counterfactuals)]

    def _calculate_feature_order(self):
        self.feature_order, self.feature_importances = calc_feature_order(self.shap_values)

    def _calculate_counterfactuals(self):
        self.counterfactuals = {}
        if np.sum(self.highlight) == 0 or self.diff_class is None:
            return
        for feature_idx, feature in enumerate(self.shap_values.feature_names):
            cf_up = self._find_upper_counterfactual(feature)
            cf_down = self._find_lower_counterfactual(feature)
            if cf_up.is_including_class(self.diff_class):
                cf_up = None
            if cf_down.is_including_class(self.diff_class):
                cf_down = None
            if cf_down is not None or cf_up is not None:
                self.counterfactuals[feature] = (cf_down, cf_up)

    def _find_upper_counterfactual(self, feature):
        return self._find_counterfactual(feature, upwards=True)

    def _find_lower_counterfactual(self, feature):
        return self._find_counterfactual(feature, upwards=False)

    def _find_counterfactual(self, feature, upwards):
        feature_idx, feature_name = self.comparer.check_feature(feature)
        diff_class_idx = np.where(self.comparer.class_names == self.diff_class)[0][0]
        X_mod = self.data[self.highlight, :].copy()
        precision = self.feature_precisions[feature_idx]
        if upwards:
            sign = 1
            start = round(self.data[self.highlight, feature_idx].max(), precision)
            limit = self.root.data[:, feature_idx].max()
        else:
            sign = -1
            start = round(self.data[self.highlight, feature_idx].min(), precision)
            limit = self.root.data[:, feature_idx].min()
        step = sign * 10 ** -precision
        y_mod = None
        for value in np.arange(start, limit + step, step):
            X_mod[:, feature_idx] = round(value, precision)
            y_mod = self.comparer.predict_mclass_diff(X_mod)
            if np.all(y_mod != diff_class_idx):
                break
        class_counts = dict(pd.Series(self.comparer.class_names[y_mod]).value_counts())
        return Counterfactual(feature_name, value=X_mod[0, feature_idx], is_direction_up=upwards, outcomes=class_counts)

    def describe_counterfactuals(self, feature):
        _, feature_name = self.comparer.check_feature(feature)
        cf_down, cf_up = self.counterfactuals.get(feature_name, (None, None))
        if cf_down is not None:
            print(cf_down)
        if cf_up is not None:
            print(cf_up)

    def describe_feature_differences(self, feature):
        feature_idx, feature_name = self.comparer.check_feature(feature)
        feature_data = self.data[self.highlight, feature_idx]
        lower_bound = feature_data.min()
        upper_bound = feature_data.max()
        if lower_bound == upper_bound:
            print(f'{feature_name} == {lower_bound}', end='')
        else:
            if lower_bound > self.root.data[:, feature_idx].min():
                print(f'{lower_bound} <= ', end='')
            print(feature_name, end='')
            if upper_bound < self.root.data[:, feature_idx].max():
                print(f' <= {upper_bound}', end='')
        print()

    def filter(self, mask):
        instance_indices = self.instance_indices[mask]
        return Explanation(self.comparer, self.orig_shap_values, instance_indices, self.cluster_classes, self.orig_pred_classes,
                           self.categorical_features, self.feature_precisions, self.orig_highlight,
                           self.diff_class, self)

    def plot_feature_dependence(self, feature, classes=None, alpha=None, color=None, fill=None, focus=None):
        if focus is not None:
            fill = np.in1d(self.instance_indices, focus.instance_indices)
            counterfactuals = focus.counterfactuals
        else:
            counterfactuals = self.counterfactuals
        feature_idx, feature_name = self.comparer.check_feature(feature)
        if classes is None:
            classes = self.cluster_classes
        if color is None:
            color = self.highlight
            color_feature_name = None
        else:
            color_feature_idx, color_feature_name = self.comparer.check_feature(color)
            color = self.data[:, color_feature_idx]
        jitter = feature_name in self.categorical_features
        s = self.shap_values[:, :, classes]
        mark_x_downwards = None
        mark_x_upwards = None
        if feature_name in counterfactuals:
            cf_down, cf_up = counterfactuals[feature_name]
            if cf_down is not None:
                mark_x_downwards = cf_down.value
            if cf_up is not None:
                mark_x_upwards = cf_up.value
        plot_feature_dependencies_for_classes(s, feature, color=color, color_label=color_feature_name, fill=fill,
                                              alpha=alpha, jitter=jitter,
                                              mark_x_upwards=mark_x_upwards, mark_x_downwards=mark_x_downwards)

    def plot_outcome_differences(self):
        diff_class_idx = index_of(self.comparer.class_names, self.diff_class)
        other_classes = [class_ for class_ in self.cluster_classes if class_ != self.diff_class]
        other_class_indices = [index_of(self.comparer.class_names, class_) for class_ in other_classes]
        s = self.shap_values
        log_odds_per_class = s.base_values + s.values.sum(1)
        log_odds_of_diff_class = log_odds_per_class[:, diff_class_idx]
        log_odds_of_other_classes = log_odds_per_class[:, other_class_indices]
        log_odds_diff = log_odds_of_other_classes - np.array([log_odds_of_diff_class for _ in other_class_indices]).T

        classes = np.concatenate([np.repeat(class_, len(log_odds_diff)) for class_ in other_classes])
        class_labels = np.concatenate([self.pred_classes for _ in other_class_indices])
        df = pd.DataFrame([classes, log_odds_diff.T.flatten(), class_labels],
                          index=['class', 'difference', 'actual class']).T
        chart = sns.catplot(x='class', y='difference', hue='actual class', data=df, alpha=0.6, aspect=2, legend=False)
        chart.axes[0][0].axhline(y=0, color='black', linewidth=1, alpha=0.5)
        plt.legend(loc='upper right')
        plt.show()
        diff_ranges_of_diff_class = [f'{round(col.min(), 2)} to {round(col.max(), 2)}'
                                     for col in log_odds_diff[self.highlight].T]
        print(dict(zip(other_classes, diff_ranges_of_diff_class)))

    def get_parent(self, n=1):
        if self.parent is None or n == 0:
            return self
        return self.parent.get_parent(n - 1)

    def describe_feature(self, feature):
        feature_idx, feature_name = self.comparer.check_feature(feature)
        feature_data = self.data[:, feature_idx]
        root_feature_data = self.root.data[:, feature_idx]
        if feature_name in self.categorical_features:
            return pd.DataFrame([
                root_feature_data.value_counts(),
                feature_data.value_counts(),
                feature_data[self.highlight].value_counts()
            ], index=['global', 'local-all', 'local-diff'])
        else:
            return pd.DataFrame(index=['global', 'local-all', 'local-diff'], data=[
                root_feature_data,
                feature_data,
                feature_data[self.highlight]]).T.describe()


class ExplanationNode(Explanation):
    def __init__(self, comparer, orig_shap_values, cluster_node, instance_indices, cluster_classes, orig_pred_classes,
                 distance_matrix, linkage_matrix, categorical_features, feature_precisions, orig_highlight,
                 diff_class=None, parent=None):
        super(ExplanationNode, self).__init__(comparer, orig_shap_values, instance_indices, cluster_classes,
                                              orig_pred_classes, categorical_features, feature_precisions,
                                              orig_highlight, diff_class, parent)
        self.cluster_node = cluster_node
        self.distance_matrix = distance_matrix
        self.linkage_matrix = linkage_matrix
        self.left = None
        self.right = None

    @property
    def distance(self):
        """Distance for this cluster in the distance matrix (metric='sqeuclidean')"""
        return self.cluster_node.dist

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
        if self.left is None:
            self.left = self._make_child(self.cluster_node.get_left())
        return self.left

    def _make_child(self, cluster_node):
        instance_indices = np.array(cluster_node.pre_order())
        return ExplanationNode(self.comparer, self.orig_shap_values, cluster_node, instance_indices, self.cluster_classes,
                               self.orig_pred_classes, self.distance_matrix, self.linkage_matrix, self.categorical_features,
                               self.feature_precisions, self.orig_highlight, self.diff_class, self)

    def get_right(self):
        if self.right is None:
            self.right = self._make_child(self.cluster_node.get_right())
        return self.right

    def get_last_child_before_diff_class_split(self):
        if not self.diff_class in list(self.class_counts.keys()):
            raise Exception('Difference class not present in cluster')
        if len(self.shap_values) == 1:
            return self
        left = self.get_left()
        right = self.get_right()
        if not self.diff_class in list(left.class_counts.keys()):
            return right.get_last_child_before_diff_class_split()
        if not self.diff_class in list(right.class_counts.keys()):
            return left.get_last_child_before_diff_class_split()
        return self

    def plot_dendrogram(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        hierarchy.dendrogram(self.linkage_matrix, orientation='right', ax=ax, no_labels=True)
        ax.set_title('Dendrogram')
        plt.show()


def make_clustering(comparer, shap_values, diff_class=None, cluster_classes=None, categorical_features=None,
                    feature_precisions=None):
    if cluster_classes is None:
        cluster_classes = shap_values.output_names
    if categorical_features is None:
        categorical_features = []
    if feature_precisions is None:
        feature_precisions = [0 for _ in range(len(comparer.feature_names))]
    values = shap_values[:, :, cluster_classes].values
    values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    D = distance.pdist(values, metric='sqeuclidean')
    Z = hierarchy.complete(D)
    cluster_root = hierarchy.to_tree(Z)
    instance_indices = np.array(cluster_root.pre_order())
    pred_classes = comparer.predict_mclass_diff(shap_values.data)
    pred_classes = comparer.class_names[pred_classes]
    if diff_class is not None:
        highlight = pred_classes == diff_class
    else:
        highlight = comparer.predict_bin_diff(shap_values.data)
    node = ExplanationNode(comparer, shap_values, cluster_root, instance_indices, cluster_classes, pred_classes, D, Z,
                           categorical_features, feature_precisions, highlight, diff_class)
    return node


