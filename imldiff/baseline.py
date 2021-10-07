import numpy as np


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
