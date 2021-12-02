from sklearn.tree import _tree, plot_tree
import numpy as np
import matplotlib.pyplot as plt
from comparers import plot_decision_boundary


def _remove_occurences(l, s):
    l = list(l)
    previous_occurences = [p.startswith(s) for p in l]
    if any(previous_occurences):
        idx = np.where(previous_occurences)[0][0]
        del l[idx]
    return l


def print_rules(tree, feature_names, class_names, focus_class=None, feature_order=None, precision=0, X_test=None, y_test=None):
    """Adapted from: https://mljar.com/blog/extract-rules-decision-tree/"""

    if feature_order is None:
        feature_order = np.arange(len(feature_names))

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
            p1 += [f"({name} <= {round(threshold, precision)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 = _remove_occurences(path, f"({name} >")
            p2 += [f"({name} > {round(threshold, precision)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path = sorted(path, key=lambda x: np.argwhere(feature_names[feature_order] == x.split('(')[1].split(' ')[0])[0][0] + (0.5 if '<=' in x else 0))
            path += [(tree_.value[node], tree_.n_node_samples[node], node)]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        node_id = path[-1][2]
        rule = f"node #{node_id}: if "

        for p in path[:-1]:
            if not rule.endswith(': if '):
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            current_class_idx = np.argmax(classes)
            if focus_class is not None and focus_class != class_names[current_class_idx]:
                continue
            share = round(classes[current_class_idx]/sum(classes), 3)
            n_instances = int(classes.sum())
            rule += f"class {class_names[current_class_idx]} (covers {share} of {n_instances} instances)"
        rules += [rule]

    for rule in rules:
        print(rule)


def get_node_ids_for_class(tree, class_idx):
    leaf_node_ids = np.where(tree.tree_.children_left == -1)[0]
    classes_per_node = np.array([np.argmax(node_values[0]) for node_values in tree.tree_.value])
    nodes_with_focus_class = np.where(classes_per_node == class_idx)[0]
    mask = np.in1d(nodes_with_focus_class, leaf_node_ids)
    return nodes_with_focus_class[mask]


def evaluate(tree, focus_class, focus_class_idx, X_test=None, y_test=None):
    node_ids = get_node_ids_for_class(tree, focus_class_idx)
    if X_test is not None:
        y_pred = tree.predict(X_test)
        focus_indices = np.where(y_pred == focus_class_idx)[0]
        if len(focus_indices) == 0:
            print(f'The tree model never classified any instance as {focus_class}')
            return
        focus_nodes = tree.apply(X_test[focus_indices])
        n_focus_nodes_present = np.in1d(focus_nodes, node_ids).sum()
        print(f'coverage (test set): {n_focus_nodes_present/len(focus_indices)}')

        if y_test is not None:
            focus_indices = np.where(focus_class_idx == y_test)[0]
            if len(focus_indices) == 0:
                print(f'y_test does not contain {focus_class}')
                return
            focus_nodes = tree.apply(X_test[focus_indices])
            n_focus_nodes_present = np.in1d(focus_nodes, node_ids).sum()
            print(f'coverage (test set ground truth): {n_focus_nodes_present/len(focus_indices)}')


def plot_tree_leafs_for_class(tree, tree_class_names, focus_classes, X, y, class_names, feature_names):
    if isinstance(focus_classes, str):
        focus_classes = [focus_classes]
    focus_class_indices = [np.where(tree_class_names == focus_class)[0][0] for focus_class in focus_classes]
    node_ids = [get_node_ids_for_class(tree, focus_class_idx).tolist() for focus_class_idx in focus_class_indices]
    node_ids = flatten(node_ids)
    mapping = dict(zip(node_ids, range(1, len(node_ids)+1)))

    predict_node = lambda X: np.array([mapping.get(node_id, 0) for node_id in tree.apply(X)])

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_decision_boundary(X, y,
                           feature_names=feature_names,
                           class_names=class_names,
                           predict=predict_node,
                           predict_value_names=['other'] + node_ids,
                           fig=fig, ax=ax)

def flatten(t):
    return [item for sublist in t for item in sublist]
