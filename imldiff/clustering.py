import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display

from explainers import plot_feature_effects


class Clustering:
    def __init__(self, comparer, shap_values, Z, root, diff_class):
        self.comparer = comparer
        self.root_shap_values = shap_values
        self.shap_values = shap_values
        self.Z = Z
        self.root = root
        self.node = root
        self.diff_class = diff_class
        self.parent_nodes = []
        self.traversed_nodes = []
        self._set_classes()

    def _set_classes(self):
        self.pred_classes = self.comparer.predict_mclass_diff(self.shap_values.data)
        self.pred_classes = self.comparer.class_names[self.pred_classes]
        self.classes = np.unique(self.pred_classes)

    def go_down(self):
        self.parent_nodes.append(self.node)
        self.node = self.node.get_left()
        return self._select_node()

    def _select_node(self):
        self.shap_values = self.root_shap_values[self.node.pre_order()]
        self._set_classes()
        print(f'Level {len(self.parent_nodes)}')
        print(pd.Series(self.pred_classes).value_counts())

    def get_next(self):
        self.traversed_nodes.append(self.node)
        self.node = self.parent_nodes.pop()
        if self.node.get_right() in self.traversed_nodes:
            return self.get_next()
        self.parent_nodes.append(self.node)
        self.node = self.node.get_right()
        return self._select_node()

    def plot_dendrogram(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        sp.cluster.hierarchy.dendrogram(self.Z, orientation='right', ax=ax, no_labels=True)
        ax.set_title('Dendrogram')
        plt.show()

    def plot_feature(self, feature):
        s = self.shap_values[:, [feature]][:, :, self.classes]
        highlight = self.pred_classes == self.diff_class
        plot_feature_effects(s, highlight=highlight, title=f'Highighted: {self.diff_class}', constrained_layout=True)

    def test(self, **kwargs):
        X_test = pd.DataFrame(self.shap_values.data.copy(), columns=self.shap_values.feature_names)
        y_before = pd.Series(self.comparer.class_names[self.comparer.predict_mclass_diff(X_test)])
        for key, value in kwargs.items():
            X_test[key] = value
        y_after = pd.Series(self.comparer.class_names[self.comparer.predict_mclass_diff(X_test)])
        mask = self.pred_classes == self.diff_class
        display(pd.DataFrame({'before': y_before[mask].value_counts(),
                              'after': y_after[mask].value_counts()}))


def make_clustering(comparer, shap_values, diff_class):
    s = shap_values[:, :, diff_class]
    D = sp.spatial.distance.pdist(s.values, metric='sqeuclidean')
    Z = sp.cluster.hierarchy.complete(D)
    root = sp.cluster.hierarchy.to_tree(Z)
    return Clustering(comparer, shap_values, Z, root, diff_class)
