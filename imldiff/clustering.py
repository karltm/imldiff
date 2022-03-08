import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
from explainers import plot_feature_dependencies, calc_feature_order
from util import RuleClassifier, constraint_matrix_to_rules, find_counterfactuals, \
    counterfactuals_to_constraint_matrix
from sklearn.metrics import classification_report, precision_recall_fscore_support


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
        return np.array([feature for feature in self.features_ordered if len(self.counterfactuals[feature]) > 0])

    @property
    def features_without_counterfactuals(self):
        return self.comparer.feature_names[~np.isin(self.comparer.feature_names, self.features_with_counterfactuals)]

    def _calculate_feature_order(self):
        self.feature_order, self.feature_importances = calc_feature_order(self.shap_values)

    def _calculate_counterfactuals(self):
        self.counterfactuals = dict([(feature, []) for feature in self.comparer.feature_names])
        if np.sum(self.highlight) == 0 or self.diff_class is None:
            return
        self.counterfactuals |= find_counterfactuals(self.comparer, self.data[self.highlight], self.root.data,
                                                    self.feature_precisions,
                                                    list(self.comparer.class_names).index(self.diff_class))

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

    def plot_feature_dependence(self, *features, classes=None, alpha=None, color=None, fill=None, focus=None,
                                figsize=None):
        if focus is not None:
            fill = np.in1d(self.instance_indices, focus.instance_indices)
            counterfactuals = focus.counterfactuals
        else:
            counterfactuals = self.counterfactuals
        if len(features) == 0:
            features = self.features_ordered
        features = [self.comparer.check_feature(feature)[1] for feature in features]
        if classes is None:
            classes = self.cluster_classes
        if color is None:
            color = self.highlight
            color_feature_name = None
        else:
            color_feature_idx, color_feature_name = self.comparer.check_feature(color)
            color = self.data[:, color_feature_idx]
        s = self.shap_values[:, :, classes]
        vlines = [[cf.value for cf in counterfactuals[feature]]
                  for feature in features if feature in counterfactuals]
        jitter = [feature in self.categorical_features for feature in features]
        plot_feature_dependencies(s[:, features], color=color, color_label=color_feature_name, fill=fill,
                                  alpha=alpha, jitter=jitter, vlines=vlines, figsize=figsize)

    def plot_outcome_differences(self):
        diff_class_idx = list(self.comparer.class_names).index(self.diff_class)
        other_classes = [class_ for class_ in self.cluster_classes if class_ != self.diff_class]
        other_class_indices = [list(self.comparer.class_names).index(class_) for class_ in other_classes]
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

    def rule_from_counterfactuals(self, *include_features):
        if len(include_features) == 0:
            include_features = self.features_with_counterfactuals
        constraint = self.constraint_matrix_from_counterfactuals(*include_features)
        rules = constraint_matrix_to_rules([constraint], self.comparer.feature_names, self.feature_order)
        instance_indices = self.instance_indices[self.highlight]
        return rules[0], constraint, instance_indices

    def constraint_matrix_from_counterfactuals(self, *include_features):
        if len(include_features) == 0:
            include_features = self.features_with_counterfactuals
        return counterfactuals_to_constraint_matrix(self.comparer.feature_names, self.feature_precisions,
                                                    include_features, self.counterfactuals)

    def evaluate_rules(self, *rules):
        r = RuleClassifier(self.comparer.feature_names, rules)
        y_diffclf = self.highlight
        y_expl = r.predict(self.data)
        print(classification_report(y_diffclf, y_expl))
        precisions, recalls, f1_scores, supports = precision_recall_fscore_support(y_diffclf, y_expl, labels=r.classes_)
        df = pd.DataFrame(np.array((precisions, recalls, f1_scores, supports)).T,
                          columns=['Precision', 'Recall', 'F1 Score', 'Support'],
                          index=r.classes_)
        df['Support'] = df['Support'].astype(int)
        return df
        #cm = confusion_matrix(y_diffclf, y_expl, labels=[False, True])
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        #fig, ax = plt.subplots(constrained_layout=True)
        #disp.plot(ax=ax)
        #ax.set_ylabel('difference classifier predictions')
        #ax.set_xlabel('explanation\'s predictions')
        #plt.show()


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
        self.left = self._make_child(self.cluster_node.get_left())
        self.right = self._make_child(self.cluster_node.get_right())

    def _make_child(self, cluster_node):
        if self.distance == 0.0:
            return None
        if cluster_node is None:
            return None
        instance_indices = np.array(cluster_node.pre_order())
        return ExplanationNode(self.comparer, self.orig_shap_values, cluster_node, instance_indices, self.cluster_classes,
                               self.orig_pred_classes, self.distance_matrix, self.linkage_matrix, self.categorical_features,
                               self.feature_precisions, self.orig_highlight, self.diff_class, self)

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
        return self.left

    def get_right(self):
        return self.right

    def get_last_child_before_diff_class_split(self):
        if not self.diff_class in list(self.class_counts.keys()):
            raise Exception('Difference class not present in cluster')
        if len(self.shap_values) == 1:
            return self
        left = self.get_left()
        right = self.get_right()
        if right is not None and not self.diff_class in list(left.class_counts.keys()):
            return right.get_last_child_before_diff_class_split()
        if right is not None and not self.diff_class in list(right.class_counts.keys()):
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


