from sklearn.tree import _tree
import numpy as np


def _remove_occurences(l, s):
    l = list(l)
    previous_occurences = [p.startswith(s) for p in l]
    if any(previous_occurences):
        idx = np.where(previous_occurences)[0][0]
        del l[idx]
    return l


def get_rules(tree, feature_names, class_names, class_=None, feature_order=None):
    """Adapted from: https://mljar.com/blog/extract-rules-decision-tree/"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1 = _remove_occurences(path, f"({name} <=")
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 = _remove_occurences(path, f"({name} >")
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            if feature_order is not None:
                path = sorted(path, key=lambda x: np.argwhere(feature_names[feature_order] == x.split('(')[1].split(' ')[0])[0][0] + (0.5 if '<=' in x else 0))
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            if class_ is not None and class_ != class_names[l]:
                continue
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
    return rules


def dt_feature_importance(model, normalize=True):
    """Source: https://stackoverflow.com/questions/49170296/scikit-learn-feature-importance-calculation-in-decision-trees"""

    left_c = model.tree_.children_left
    right_c = model.tree_.children_right

    impurity = model.tree_.impurity
    node_samples = model.tree_.weighted_n_node_samples

    # Initialize the feature importance, those not used remain zero
    feature_importance = np.zeros((model.tree_.n_features,))

    for idx,node in enumerate(model.tree_.feature):
        if node >= 0:
            # Accumulate the feature importance over all the nodes where it's used
            feature_importance[node]+=impurity[idx]*node_samples[idx]- \
                                      impurity[left_c[idx]]*node_samples[left_c[idx]]- \
                                      impurity[right_c[idx]]*node_samples[right_c[idx]]

    # Number of samples at the root node
    feature_importance/=node_samples[0]

    if normalize:
        normalizer = feature_importance.sum()
        if normalizer > 0:
            feature_importance/=normalizer

    return feature_importance
