from sklearn.tree import _tree
import numpy as np
import pandas as pd


def print_rules_for_binary(tree, feature_names, class_names, print_class):
    # TODO check if features are passed in the right order

    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    tree_ = tree.tree_

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            if tree_.value[child][0][0] < tree_.value[child][0][1]:
                sclass = class_names[1]
            else:
                sclass = class_names[0]

            lineage = [(sclass)]
        if child in left:
            parent = np.where(left == child)[0].item()
            lineage.append(str(features[parent]) +
                           " <= " + str(threshold[parent]))
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            lineage.append(str(features[parent]) +
                           " > " + str(threshold[parent]))
            split = 'r'

        #lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    rules = []
    for child in idx:

        rule = '{'
        for node in recurse(left, right, child):

            if str(node) == 'diff' or str(node) == 'no_diff':
                diff = str(node)
                rule = rule[:-2]
                rule += '}'
                rules.append({diff: rule})
            else:
                rule += str(node) + ', '

    for rule in rules:
        for key in rule:
            if key == print_class:
                print(rule)


def tree_to_code(tree, feature_names):
    decision_rules = []
    tree_ = tree.tree_

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth, decision_rules):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))

            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1, decision_rules)
