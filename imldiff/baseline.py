from sklearn.tree import _tree
import numpy as np


def _remove_occurences(l, s):
    l = list(l)
    previous_occurences = [p.startswith(s) for p in l]
    if any(previous_occurences):
        idx = np.where(previous_occurences)[0][0]
        del l[idx]
    return l


def print_rules(tree, feature_names, class_names, class_=None, feature_order=None, precision=3):
    rules = get_rules(tree, feature_names, class_names, class_, feature_order, precision)
    for rule in rules:
        print(rule)


def get_rules(tree, feature_names, class_names, class_=None, feature_order=None, precision=3):
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
            p1 += [f"({name} <= {np.round(threshold, precision)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 = _remove_occurences(path, f"({name} >")
            p2 += [f"({name} > {np.round(threshold, precision)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            if feature_order is not None:
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
        rule = f"node #{path[-1][2]}: if "

        for p in path[:-1]:
            if not rule.endswith(': if '):
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
            rule += f"class: {class_names[l]} ({int(classes[l])}/{int(sum(classes))} instances)"
        rules += [rule]
    return rules
