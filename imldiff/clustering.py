import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display
from explainers import plot_feature_effects_per_feature, ensure_are_shap_values


class ClusterNode:
    def __init__(self, comparer, shap_values, node, parent, diff_class=None, cluster_classes=None,
                 categorical_features=[]):
        self.comparer = comparer
        self.root_shap_values = shap_values
        self.node = node
        self.parent = parent
        self.diff_class = diff_class
        self.cluster_classes = cluster_classes
        self.shap_values = self.root_shap_values[self.node.pre_order()]
        self.categorical_features = categorical_features
        self._set_classes()
        print(self)
        print(self.class_counts)

    def _set_classes(self):
        self.is_different = self.comparer.predict_bin_diff(self.shap_values.data)
        self.pred_classes = self.comparer.predict_mclass_diff(self.shap_values.data)
        self.pred_classes = self.comparer.class_names[self.pred_classes]

    def __str__(self):
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

    def __repr__(self):
        return f'{self.__class__.__name__}({self.node.id})'

    @property
    def class_counts(self):
        return pd.Series(self.pred_classes).value_counts()

    def get_left(self):
        return ClusterNode(self.comparer, self.root_shap_values, self.node.get_left(), self, self.diff_class,
                           self.cluster_classes, self.categorical_features)

    def get_right(self):
        return ClusterNode(self.comparer, self.root_shap_values, self.node.get_right(), self, self.diff_class,
                           self.cluster_classes, self.categorical_features)

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

    @property
    def highlight(self):
        if self.diff_class is not None:
            return self.pred_classes == self.diff_class
        else:
            return self.is_different

    def plot_feature(self, feature, classes=None, alpha=None, focus=None):
        fill = None
        if classes is None:
            classes = self.cluster_classes
        if focus is not None:
            fill = np.in1d(self.node.pre_order(), focus.node.pre_order())
        jitter = self.shap_values[:, feature].feature_names in self.categorical_features
        s = ensure_are_shap_values(self.shap_values)[:, :, classes]
        plot_feature_effects_per_feature(s, feature, color=self.highlight, fill=fill, alpha=alpha, jitter=jitter)

    def test(self, **kwargs):
        X_test = pd.DataFrame(self.shap_values.data.copy(), columns=self.shap_values.feature_names)
        y_before = pd.Series(self.comparer.class_names[self.comparer.predict_mclass_diff(X_test)])
        for key, value in kwargs.items():
            X_test[key] = value
        y_after = pd.Series(self.comparer.class_names[self.comparer.predict_mclass_diff(X_test)])
        display(pd.DataFrame({'before': y_before[self.highlight].value_counts(),
                              'after': y_after[self.highlight].value_counts()}))


def make_clustering(comparer, shap_values, diff_class=None, cluster_classes=None, categorical_features=None):
    s = ensure_are_shap_values(shap_values)
    if cluster_classes is None:
        cluster_classes = s.output_names
    values = s[:, :, cluster_classes].values
    values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    D = sp.spatial.distance.pdist(values, metric='sqeuclidean')
    Z = sp.cluster.hierarchy.complete(D)
    root = sp.cluster.hierarchy.to_tree(Z)
    return ClusterNode(comparer, shap_values, root, None, diff_class, cluster_classes, categorical_features), Z


def plot_dendrogram(Z):
    fig, ax = plt.subplots(figsize=(7, 7))
    sp.cluster.hierarchy.dendrogram(Z, orientation='right', ax=ax, no_labels=True)
    ax.set_title('Dendrogram')
    plt.show()
