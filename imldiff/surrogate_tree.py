import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree._tree import Tree, TREE_UNDEFINED
from util import plot_decision_boundary, constraint_matrix_to_rules, evaluate, RuleClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support


def train_surrogate_tree(X, y, max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth)
    return model.fit(X, y)


def get_feature_importances(model):
    feature_importances = model.feature_importances_
    feature_order = np.flip(np.argsort(feature_importances))
    return feature_importances, feature_order


def plot_feature_importances(feature_names, feature_importances, feature_order, figsize=(5, 2)):
    df = pd.DataFrame({
        'Features': feature_names,
        'Importance': feature_importances,
    })
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='Importance', y='Features', data=df, order=feature_names[feature_order], ax=ax)


def plot_surrogate_tree(model, feature_names, class_names=None, precision=3, figsize=(14, 14), node_ids=False):
    if class_names is None:
        class_names = [str(label) for label in model.classes_]
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(model, feature_names=feature_names, ax=ax, precision=precision,
              class_names=np.array(class_names)[model.classes_], node_ids=node_ids)


def plot_tree_leafs_for_class(tree, tree_class_names, focus_classes, X, y, class_names, feature_names,
                              feature_x=0, feature_y=1, figsize=(7, 7)):
    if isinstance(focus_classes, str):
        focus_classes = [focus_classes]
    if isinstance(feature_x, str):
        feature_x = list(feature_names).index(feature_x)
    if isinstance(feature_y, str):
        feature_y = list(feature_names).index(feature_y)
    focus_class_indices = [list(tree_class_names).index(focus_class) for focus_class in focus_classes]
    node_ids = [get_node_ids_for_class(tree, focus_class_idx).tolist() for focus_class_idx in focus_class_indices]
    node_ids = _flatten(node_ids)
    mapping = dict(zip(node_ids, range(1, len(node_ids)+1)))

    predict_node = lambda X: np.array([mapping.get(node_id, 0) for node_id in tree.apply(X)])

    fig, ax = plt.subplots(figsize=figsize)
    plot_decision_boundary(X, y, idx_x=feature_x, idx_y=feature_y,
                           feature_names=feature_names,
                           class_names=class_names,
                           predict=predict_node,
                           predict_value_names=['other'] + node_ids,
                           fig=fig, ax=ax)


def _flatten(t):
    return [item for sublist in t for item in sublist]


def get_node_ids_for_class(tree, class_idx):
    leaf_node_ids = np.where(tree.tree_.children_left == -1)[0]
    classes_per_node = np.array([np.argmax(node_values[0]) for node_values in tree.tree_.value])
    nodes_with_focus_class = np.where(classes_per_node == class_idx)[0]
    mask = np.in1d(nodes_with_focus_class, leaf_node_ids)
    return nodes_with_focus_class[mask]


def extract_rules(model, feature_names, classes_to_include, X, y):
    constraints, class_occurences, labels, node_ids = tree_to_constraint_matrix(model)
    rules = constraint_matrix_to_rules(constraints, feature_names)

    mask = np.in1d(labels, classes_to_include)
    rules = np.array(rules)[mask].tolist()
    class_occurences = class_occurences[mask, :]
    labels = labels[mask]
    constraints = constraints[mask]
    node_ids = np.array(node_ids)[mask]

    rule_order = np.flip(class_occurences.sum(1).argsort())
    rules = np.array(rules)[rule_order].tolist()
    class_occurences = class_occurences[rule_order]
    labels = labels[rule_order]
    constraints = constraints[rule_order]
    node_ids = np.array(node_ids)[rule_order]

    pred_node_ids = model.apply(X)
    instance_indices_per_rule = []
    for node_id, label in zip(node_ids, labels):
        mask = (pred_node_ids == node_id) & (y == label)
        indices = np.where(mask)[0]
        instance_indices_per_rule.append(indices)

    return constraints, rules, class_occurences, labels, instance_indices_per_rule


def tree_to_rules(tree: DecisionTreeClassifier, feature_names, feature_order=None):
    constraints, class_occurences, labels, node_ids = tree_to_constraint_matrix(tree)
    rules = constraint_matrix_to_rules(constraints, feature_names, feature_order)
    return rules, class_occurences, labels, node_ids


def tree_to_constraint_matrix(tree: DecisionTreeClassifier):
    constraints, class_occurences, node_ids = _recurse(tree.tree_)
    constraints = np.array(constraints)
    class_occurences = np.array(class_occurences)
    labels = class_occurences.argmax(1)
    labels = tree.classes_[labels]
    return constraints, class_occurences, labels, node_ids


def _recurse(tree: Tree, constraint=None, node_id=0):
    if constraint is None:
        constraint = np.full((tree.n_features, 2), np.nan)
    feature_idx = tree.feature[node_id]
    if feature_idx == TREE_UNDEFINED:
        class_occurences = tree.value[node_id][0]
        return [constraint], [class_occurences], [node_id]

    threshold = tree.threshold[node_id]
    left_constraint = constraint.copy()
    left_constraint[feature_idx, 1] = threshold
    right_constraint = constraint.copy()
    right_constraint[feature_idx, 0] = threshold
    left_constraints, left_class_occurences, left_node_ids = \
        _recurse(tree, left_constraint, tree.children_left[node_id])
    right_constraints, right_class_occurences, right_node_ids = \
        _recurse(tree, right_constraint, tree.children_right[node_id])
    return left_constraints + right_constraints,\
           left_class_occurences + right_class_occurences,\
           left_node_ids + right_node_ids


def print_rules(rules, class_occurences, class_names=None, labels=None):
    for idx, (rule, class_occurences, label) in enumerate(zip(rules, class_occurences, labels), 1):
        print(f'{idx}. {rule} => {class_names[label]} {class_occurences.astype(int).tolist()}')


def search_max_depth_parameter(X, y_true, feature_names, class_names, start=2, stop=None, X_test=None, y_test=None):
    classes = np.arange(len(class_names))
    max_depth = start
    parameters = []
    metrics = []
    while stop is None or max_depth <= stop:
        model = train_surrogate_tree(X, y_true, max_depth=max_depth)
        constraints, rules, _, labels, _ = extract_rules(model, feature_names, classes, X, y_true)
        results = evaluate(model, X_test, y_test, class_names)
        for label in np.unique(labels):
            result = results.loc[class_names[label]]
            mask = labels == label
            n_rules = np.sum(mask)
            n_constraints = np.sum(~np.isnan(constraints[mask]))
            parameters.append(max_depth)
            metrics.append((class_names[label], result['Precision'], result['Recall'], result['F1 Score'], n_rules, n_constraints))

        if max_depth > model.get_depth():
            break

        max_depth += 1

    df = pd.DataFrame(metrics, columns=['Label', 'Precision', 'Recall', 'F1 Score', 'Rules', 'Constraints'])
    df.insert(0, 'Depth', parameters)
    return df
