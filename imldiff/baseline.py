from sklearn.tree import _tree
import numpy as np


def _remove_occurences(l, s):
    l = list(l)
    previous_occurences = [p.startswith(s) for p in l]
    if any(previous_occurences):
        idx = np.where(previous_occurences)[0][0]
        del l[idx]
    return l


def print_rules(tree, feature_names, class_names, focus_class=None, feature_order=None, precision=0, X_test=None):
    """Adapted from: https://mljar.com/blog/extract-rules-decision-tree/"""

    if feature_order is None:
        feature_order = np.arange(len(feature_names))

    focus_class_idx = np.where(class_names == focus_class)[0][0]

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
            p1 += [f"({name} <= {round_down(threshold, precision)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 = _remove_occurences(path, f"({name} >")
            p2 += [f"({name} > {round_down(threshold, precision)})"]
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

    n_instances_covered = 0
    n_instances_total = 0
    rules = []
    node_ids = []
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
            n_instances_total += classes[focus_class_idx]
            if focus_class is not None and focus_class != class_names[current_class_idx]:
                continue
            share = round(classes[current_class_idx]/sum(classes), 3)
            n_instances = int(classes.sum())
            rule += f"class {class_names[current_class_idx]} (covers {share} of {n_instances} instances)"
            n_instances_covered += classes[current_class_idx]
        node_ids.append(node_id)
        rules += [rule]

    for rule in rules:
        print(rule)

    if focus_class is not None:
        print(f'coverage (train set): {n_instances_covered / n_instances_total}')

    if X_test is not None:
        y_pred = tree.predict(X_test)
        focus_indices = np.where(y_pred)[0]
        focus_nodes = tree.apply(X_test[focus_indices])
        n_focus_nodes_present = np.in1d(focus_nodes, node_ids).sum()
        print(f'coverage (test set): {n_focus_nodes_present/len(focus_indices)}')


def round_down(value, decimals):
    factor = 10 ** decimals
    return np.floor(value * factor) / factor
