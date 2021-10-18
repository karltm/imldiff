import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display

from explainers import plot_feature_effects_per_feature


class Clustering:
    def __init__(self, comparer, shap_values, Z, root, diff_class, cluster_classes):
        self.comparer = comparer
        self.root_shap_values = shap_values
        self.shap_values = shap_values
        self.Z = Z
        self.root = root
        self.node = root
        self.diff_class = diff_class
        self.cluster_classes = cluster_classes
        self.parent_nodes = []
        self.traversed_nodes = []
        self._select_node()

    def _set_classes(self):
        self.pred_classes = self.comparer.predict_mclass_diff(self.shap_values.data)
        self.pred_classes = self.comparer.class_names[self.pred_classes]
        self.classes = np.unique(self.pred_classes)

    def go_down(self):
        if not self._is_visited(self.node.left):
            return self.go_left()
        elif not self._is_visited(self.node.right):
            return self.go_right()
        raise Exception('Child nodes already visited')

    def _is_visited(self, node):
        return node.id in [node.id for node in self.traversed_nodes]

    def go_left(self):
        self.parent_nodes.append(self.node)
        self.node = self.node.get_left()
        return self._select_node()

    def go_right(self):
        self.parent_nodes.append(self.node)
        self.node = self.node.get_right()
        return self._select_node()

    def _select_node(self):
        self.traversed_nodes.append(self.node)
        self.shap_values = self.root_shap_values[self.node.pre_order()]
        self._set_classes()
        print(f'node #{self.node.id} (level {len(self.parent_nodes)})')
        print(pd.Series(self.pred_classes).value_counts())

    def go_up(self):
        self.node = self.parent_nodes.pop()
        return self._select_node()

    def get_next(self):
        self.go_up()
        if self._is_visited(self.node.right):
            return self.get_next()
        return self.go_right()

    def plot_dendrogram(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        sp.cluster.hierarchy.dendrogram(self.Z, orientation='right', ax=ax, no_labels=True)
        ax.set_title('Dendrogram')
        plt.show()

    def describe_feature(self, feature):
        return pd.DataFrame(index=['global', 'local-all', 'local-diff'], data=[
            self.root_shap_values[:, feature].data,
            self.shap_values[:, feature].data,
            self.shap_values[self.pred_classes == self.diff_class, feature].data]).T.describe()

    def plot_feature(self, feature, classes=None, alpha=None):
        if classes is None:
            classes = self.classes
        s = self.shap_values[:, :, classes]
        plot_feature_effects_per_feature(s, feature, highlight=self.pred_classes == self.diff_class, alpha=alpha)

    def test(self, **kwargs):
        X_test = pd.DataFrame(self.shap_values.data.copy(), columns=self.shap_values.feature_names)
        y_before = pd.Series(self.comparer.class_names[self.comparer.predict_mclass_diff(X_test)])
        for key, value in kwargs.items():
            X_test[key] = value
        y_after = pd.Series(self.comparer.class_names[self.comparer.predict_mclass_diff(X_test)])
        mask = self.pred_classes == self.diff_class
        display(pd.DataFrame({'before': y_before[mask].value_counts(),
                              'after': y_after[mask].value_counts()}))


def make_clustering(comparer, shap_values, diff_class, *add_cluster_classes):
    classes = list(add_cluster_classes) + [diff_class]
    s = shap_values[:, :, classes]
    values = s.values
    values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    D = sp.spatial.distance.pdist(values, metric='sqeuclidean')
    Z = sp.cluster.hierarchy.complete(D)
    root = sp.cluster.hierarchy.to_tree(Z)
    return Clustering(comparer, shap_values, Z, root, diff_class, classes)
