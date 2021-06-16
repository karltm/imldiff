import numpy as np
from scipy.special import logsumexp
import shap
from shap.utils import hclust_ordering
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
    
def reduce_multiclass_proba_diff_shap_values(shap_values):
    """
    Reduce multiclass difference SHAP-values to binary class difference SHAP-values
    assuming they where generated on probability (or 0/1) estimates of the base models
    """
    difference_classes = get_difference_classes(n_classes=shap_values.values.shape[2])
    new_values = np.sum(shap_values.values[:, :, difference_classes], axis=2)
    new_base_values = np.sum(shap_values.base_values[:, difference_classes], axis=1)
    output_names = None
    output_indexes = None
    return shap.Explanation(new_values, new_base_values,
                            shap_values.data,
                            shap_values.display_data,
                            shap_values.instance_names,
                            shap_values.feature_names,
                            output_names,
                            output_indexes,
                            shap_values.main_effects,
                            shap_values.hierarchical_values,
                            shap_values.clustering)

def get_difference_classes(n_classes):
    classes = np.arange(n_classes)
    equality_classes = get_equality_classes(n_classes)
    return np.setdiff1d(classes, equality_classes)

def get_equality_classes(n_classes):
    n_base_classes = np.sqrt(n_classes).astype(int)
    class_matrix = np.arange(n_classes).reshape((n_base_classes, n_base_classes))
    return np.diagonal(class_matrix)

def encode_one_hot(labels, classes):
    indices = np.searchsorted(classes, labels)
    return np.eye(len(classes))[indices]

def calc_binary_log_odds_from_log_proba(log_proba):
    return log_proba[:, 1] - log_proba[:, 0]

def calc_log_odds_from_log_proba(log_proba):
    log_odds = np.empty(log_proba.shape)
    for i in range(log_proba.shape[1]):
        class_mask = [True] * log_proba.shape[1]
        class_mask[i] = False
        log_odds[:, i] = log_proba[:, i] - logsumexp(log_proba[:, class_mask], axis=1)
    return log_odds


def calibrate_classifier(est, name, X_train, X_test, y_train, y_test, cv=10, fig_index=1):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=cv, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=cv, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic (Baseline)'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=max(y_train.max(), y_test.max()))
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

    return isotonic, sigmoid


def calc_feature_order(shap_values):
    if len(shap_values.shape) == 2:
        values = np.abs(shap_values.values).mean(axis=0)
    elif len(shap_values.shape) == 3:
        values = np.abs(shap_values.values).mean(axis=2).mean(axis=0)
    else:
        raise Exception(f'invalid dimensions: {shap_values.shape}')
    feature_order = np.flip(values.argsort())
    feature_importance = shap.Explanation(values, feature_names=shap_values.feature_names)
    return feature_order, feature_importance


def calc_class_order(shap_values, information_threshold_pct=0.9):
    if not len(shap_values.shape) == 3:
        raise Exception('only multiclass kinds allowed')
    class_importances = np.abs(shap_values.values).mean(axis=1).mean(axis=0)
    class_order = np.flip(np.argsort(class_importances))
    class_importances_cumulated = np.cumsum(class_importances[class_order])
    total_importance = np.sum(class_importances)
    proportional_importances = class_importances_cumulated / total_importance
    n_informative_classes = 1 + np.where(proportional_importances > information_threshold_pct)[0][0]
    return class_order, class_importances, n_informative_classes
    # TODO: plot results


def calc_instance_order(shap_values):
    values = shap_values.values
    if len(values.shape) == 3:
        values = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
    instance_order = np.argsort(hclust_ordering(values))
    return instance_order
    # informative_class_indices = class_order[:n_informative_classes]


