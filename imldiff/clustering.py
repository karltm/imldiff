import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display
from explainers import plot_feature_effects_per_feature, ensure_are_shap_values, calc_feature_order


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
    def __init__(self, comparer, root_shap_values, shap_values, diff_class=None, feature_precisions=None,
                 cluster_classes=None, categorical_features=None):
        self.comparer = comparer
        self.root_shap_values = root_shap_values
        self.shap_values = shap_values
        self.diff_class = diff_class
        self.feature_precisions = feature_precisions
        if self.feature_precisions is None:
            self.feature_precisions = [0 for _ in range(len(self.comparer.feature_names))]
        self.cluster_classes = cluster_classes
        if categorical_features is not None:
            self.categorical_features = categorical_features
        else:
            self.categorical_features = []
        self._set_classes()
        self._calculate_counterfactuals()
        self._calculate_feature_order()

    @property
    def highlight(self):
        if self.diff_class is not None:
            return self.pred_classes == self.diff_class
        else:
            return self.is_different

    @property
    def feature_names_ordered(self):
        return self.comparer.feature_names[self.feature_order]

    @property
    def feature_names_relevant(self):
        mask = np.isin(self.feature_names_ordered, list(self.counterfactuals.keys()))
        return self.feature_names_ordered[mask]

    @property
    def class_counts(self):
        return pd.Series(self.pred_classes).value_counts()

    def _set_classes(self):
        self.is_different = self.comparer.predict_bin_diff(self.shap_values.data)
        self.pred_classes = self.comparer.predict_mclass_diff(self.shap_values.data)
        self.pred_classes = self.comparer.class_names[self.pred_classes]

    def _calculate_counterfactuals(self):
        self.counterfactuals = {}
        if np.sum(self.highlight) == 0 or self.diff_class is None:
            return
        for feature_idx, feature in enumerate(self.shap_values.feature_names):
            cf_up = self.find_cf_up(feature)
            cf_down = self.find_cf_down(feature)
            if cf_up.is_including_class(self.diff_class):
                cf_up = None
            if cf_down.is_including_class(self.diff_class):
                cf_down = None
            if cf_down is not None or cf_up is not None:
                self.counterfactuals[feature] = (cf_down, cf_up)

    def _calculate_feature_order(self):
        self.feature_order, self.feature_importances = calc_feature_order(self.shap_values)

    def find_cf_up(self, feature):
        return self._find_cf(feature, upwards=True)

    def find_cf_down(self, feature):
        return self._find_cf(feature, upwards=False)

    def _find_cf(self, feature, upwards):
        if isinstance(feature, str):
            feature_idx = np.where(self.comparer.feature_names == feature)[0][0]
        else:
            feature_idx = feature
            feature = self.comparer.feature_names[feature_idx]
        diff_class_idx = np.where(self.comparer.class_names == self.diff_class)[0][0]
        s = self.shap_values[self.highlight]
        X_mod = s.data.copy()
        precision = self.feature_precisions[feature_idx]
        if upwards:
            sign = 1
            start = round(self.shap_values[self.highlight, feature].data.max(), precision)
            limit = self.root_shap_values[:, feature].data.max()
        else:
            sign = -1
            start = round(self.shap_values[self.highlight, feature].data.min(), precision)
            limit = self.root_shap_values[:, feature].data.min()
        step = sign * 10 ** -precision
        y_mod = None
        for value in np.arange(start, limit + step, step):
            X_mod[:, feature_idx] = round(value, precision)
            y_mod = self.comparer.predict_mclass_diff(X_mod)
            if np.all(y_mod != diff_class_idx):
                break
        class_counts = dict(pd.Series(self.comparer.class_names[y_mod]).value_counts())
        return Counterfactual(feature, value=X_mod[0, feature_idx], is_direction_up=upwards, outcomes=class_counts)

    def describe_counterfactuals(self, feature):
        cf_down, cf_up = self.counterfactuals[feature]
        if cf_down is not None:
            print(cf_down)
        if cf_up is not None:
            print(cf_up)

    def describe_feature_differences(self, feature):
        data = self.shap_values[self.highlight, feature].data
        lower_bound = data.min()
        upper_bound = data.max()
        if lower_bound == upper_bound:
            print(f'{feature} == {lower_bound}', end='')
        else:
            if lower_bound > self.root_shap_values[:, feature].data.min():
                print(f'{lower_bound} <= ', end='')
            print(feature, end='')
            if upper_bound < self.root_shap_values[:, feature].data.max():
                print(f' <= {upper_bound}', end='')
        print()

    def filter(self, mask):
        return Explanation(self.comparer, self.root_shap_values, self.shap_values[mask], self.diff_class,
                           self.feature_precisions, self.cluster_classes, self.categorical_features)

    def test(self, **kwargs):
        X_test = pd.DataFrame(self.shap_values.data.copy(), columns=self.shap_values.feature_names)
        y_before = pd.Series(self.comparer.class_names[self.comparer.predict_mclass_diff(X_test)])
        for key, value in kwargs.items():
            X_test[key] = value
        y_after = pd.Series(self.comparer.class_names[self.comparer.predict_mclass_diff(X_test)])
        display(pd.DataFrame({'before': y_before[self.highlight].value_counts(),
                              'after': y_after[self.highlight].value_counts()}))

    def plot_feature(self, feature, classes=None, alpha=None, color=None, fill=None, counterfactuals=None):
        if isinstance(feature, str):
            feature_idx = np.where(self.comparer.feature_names == feature)[0][0]
        else:
            feature_idx = feature
            feature = self.comparer.feature_names[feature_idx]
        if classes is None:
            classes = self.cluster_classes
        if color is None:
            color = self.highlight
        color_feature = None
        if isinstance(color, int):
            color_feature = self.shap_values.feature_names[color]
        if isinstance(color, str):
            color_feature = color
        if color_feature is not None:
            color = self.shap_values[:, color_feature].data
        if counterfactuals is None:
            counterfactuals = self.counterfactuals
        jitter = self.shap_values[:, feature].feature_names in self.categorical_features
        s = ensure_are_shap_values(self.shap_values)[:, :, classes]
        mark_x_downwards = None
        mark_x_upwards = None
        if feature in counterfactuals:
            cf_down, cf_up = counterfactuals[feature]
            if cf_down is not None:
                mark_x_downwards = cf_down.value
            if cf_up is not None:
                mark_x_upwards = cf_up.value
        plot_feature_effects_per_feature(s, feature, color=color, color_label=color_feature, fill=fill, alpha=alpha,
                                         jitter=jitter, mark_x_upwards=mark_x_upwards, mark_x_downwards=mark_x_downwards)


class ClusterNode(Explanation):
    def __init__(self, comparer, shap_values, node, parent, diff_class=None, cluster_classes=None,
                 categorical_features=None, feature_precisions=None):
        super(ClusterNode, self).__init__(comparer, shap_values, shap_values[node.pre_order()], diff_class, feature_precisions, cluster_classes, categorical_features)
        self.node = node
        self.parent = parent
        self.root = self if parent is None else parent.root

    @property
    def distance(self):
        """Distance for this cluster in the distance matrix (metric='sqeuclidean')"""
        return self.node.dist

    def __repr__(self):
        if self.parent is None:
            return 'root'
        if self.parent.node.get_left().id == self.node.id:
            name = 'L'
        else:
            name = 'R'
        parent_name = str(self.parent)
        if parent_name == 'root':
            return name
        else:
            return parent_name + name

    def get_left(self):
        return ClusterNode(self.comparer, self.root_shap_values, self.node.get_left(), self, self.diff_class,
                           self.cluster_classes, self.categorical_features, self.feature_precisions)

    def get_right(self):
        return ClusterNode(self.comparer, self.root_shap_values, self.node.get_right(), self, self.diff_class,
                           self.cluster_classes, self.categorical_features, self.feature_precisions)

    def get(self, name):
        if name.startswith('L'):
            return self.get_left().get(name[1:])
        elif name.startswith('R'):
            return self.get_right().get(name[1:])
        else:
            return self

    def describe_feature(self, feature):
        s = self.shap_values[:, feature]
        if s.feature_names in self.categorical_features:
            return pd.DataFrame([
                    pd.Series(self.root_shap_values[:, feature].data).value_counts(),
                    pd.Series(s.data).value_counts(),
                    pd.Series(s[self.highlight].data).value_counts()
                ], index=['global', 'local-all', 'local-diff'])
        else:
            return pd.DataFrame(index=['global', 'local-all', 'local-diff'], data=[
                self.root_shap_values[:, feature].data,
                s.data,
                s[self.highlight].data]).T.describe()

    def plot_feature(self, feature, classes=None, alpha=None, focus=None, color=None):
        fill = None
        if focus is not None:
            fill = np.in1d(self.node.pre_order(), focus.node.pre_order())
        counterfactuals = focus.counterfactuals if focus is not None else self.counterfactuals
        return super(ClusterNode, self).plot_feature(feature, classes, alpha, color, fill, counterfactuals)


def make_clustering(comparer, shap_values, diff_class=None, cluster_classes=None, categorical_features=None,
                    feature_precisions=None):
    s = ensure_are_shap_values(shap_values)
    if cluster_classes is None:
        cluster_classes = s.output_names
    values = s[:, :, cluster_classes].values
    values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    D = sp.spatial.distance.pdist(values, metric='sqeuclidean')
    Z = sp.cluster.hierarchy.complete(D)
    root = sp.cluster.hierarchy.to_tree(Z)
    node = ClusterNode(comparer, shap_values, root, None, diff_class, cluster_classes, categorical_features, feature_precisions)
    return node, Z


def plot_dendrogram(Z):
    fig, ax = plt.subplots(figsize=(7, 7))
    sp.cluster.hierarchy.dendrogram(Z, orientation='right', ax=ax, no_labels=True)
    ax.set_title('Dendrogram')
    plt.show()
