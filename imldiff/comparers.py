from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from difference_models import BinaryDifferenceClassifier, MulticlassDifferenceClassifier
from scipy.special import logsumexp
from util import get_index_and_name, plot_decision_boundary


class ModelComparer:
    """ Helper class for comparing two models """
    
    def __init__(self, clf_a, clf_b, feature_names):
        self.feature_names = np.array(feature_names)
        self.clf_a = clf_a
        self.clf_b = clf_b
        self.bin_diff_clf = BinaryDifferenceClassifier(clf_a, clf_b)
        self.mclass_diff_clf = MulticlassDifferenceClassifier(clf_a, clf_b)
        self.classifier_names = np.array(['A', 'B'])
        self.bin_class_names = np.array(['equal', 'different'])
        
    def fit(self, X, y):
        return self

    @property
    def has_log_odds_support(self):
        return hasattr(self.clf_a, 'predict_log_proba') and hasattr(self.clf_b, 'predict_log_proba')

    @property
    def has_probability_support(self):
        return hasattr(self.clf_a, 'predict_proba') and hasattr(self.clf_b, 'predict_proba')

    @property
    def classifiers(self):
        return self.clf_a, self.clf_b

    @property
    def base_classes(self):
        return self.mclass_diff_clf.base_classes_

    @property
    def base_class_names(self):
        return np.array([str(label) for label in self.base_classes])

    @property
    def combined_class_names(self):
        return np.array([f'{clf_name}.{class_name}'
                         for clf_name in self.classifier_names
                         for class_name in self.base_class_names])

    @property
    def is_binary_classification_task(self):
        return len(self.base_classes) == 2

    @property
    def classes(self):
        return self.mclass_diff_clf.classes_

    @property
    def class_tuples(self):
        return self.mclass_diff_clf.class_tuples_

    @property
    def class_names(self):
        return np.array([str(label) for label in self.class_tuples])
    
    @property
    def difference_classes(self):
        return self.mclass_diff_clf.difference_classes_

    @property
    def difference_class_names(self):
        return self.class_names[self.difference_classes]
    
    @property
    def equality_classes(self):
        return self.mclass_diff_clf.equality_classes_

    @property
    def equality_class_names(self):
        return self.class_names[self.equality_classes]

    def predict_combined_oh_encoded(self, X):
        y_pred_a = _encode_one_hot(self.clf_a.predict(X), self.base_classes)
        y_pred_b = _encode_one_hot(self.clf_b.predict(X), self.base_classes)
        return np.concatenate([y_pred_a, y_pred_b], axis=1)

    def predict_combined_proba(self, X):
        y_pred_a = self.clf_a.predict_proba(X)
        y_pred_b = self.clf_b.predict_proba(X)
        return np.concatenate([y_pred_a, y_pred_b], axis=1)

    def predict_combined_log_odds(self, X):
        y_pred_a = _calc_log_odds_from_log_proba(self.clf_a.predict_log_proba(X))
        y_pred_b = _calc_log_odds_from_log_proba(self.clf_b.predict_proba(X))
        return np.concatenate([y_pred_a, y_pred_b], axis=1)

    def predict_bin_diff(self, X):
        return self.bin_diff_clf.predict(X)

    def predict_bin_diff_proba(self, X):
        return self.bin_diff_clf.predict_proba(X)[:, 1]

    def predict_bin_diff_log_odds(self, X):
        return _calc_binary_log_odds_from_log_proba(self.bin_diff_clf.predict_log_proba(X))

    def predict_mclass_diff(self, X):
        return self.mclass_diff_clf.predict(X)

    def predict_mclass_diff_oh_encoded(self, X):
        return _encode_one_hot(self.mclass_diff_clf.predict(X), self.classes)

    def predict_mclass_diff_proba(self, X):
        return self.mclass_diff_clf.predict_proba(X)

    def predict_mclass_diff_log_odds(self, X):
        return _calc_log_odds_from_log_proba(self.mclass_diff_clf.predict_log_proba(X))

    def plot_individual_clf_decision_boundaries(self, X, X_display=None, y_true=None, separate=False,
                                                kind='label', idx_x=0, idx_y=1, zlim=None, figsize=None, **kwargs):
        class_names = None
        if kind == 'label':
            class_names = self.mclass_diff_clf.base_classes_
            if not separate:
                if figsize is None:
                    figsize = (2*7, 7)
                fig, axs = plt.subplots(ncols=2, sharey=True, figsize=figsize, constrained_layout=True)
                for clf_name, predict, ax in zip(self.classifier_names,
                                                 [self.clf_a.predict, self.clf_b.predict],
                                                 axs):
                    plot_decision_boundary(X, y_true, clf_name, self.feature_names, X_display,
                                           predict=predict, class_names=class_names, zlim=zlim,
                                           idx_x=idx_x, idx_y=idx_y,
                                           fig=fig, ax=ax, **kwargs)
            else:
                if figsize is None:
                    figsize = (2*7, len(self.base_classes)*7)
                fig, axs = plt.subplots(nrows=len(self.base_classes), ncols=2, sharex=True, sharey=True,
                                        figsize=figsize, constrained_layout=True, squeeze=False)
                
                for clf_name, predict, axs_row in zip(self.classifier_names,
                                                        [self.clf_a.predict, self.clf_b.predict],
                                                        axs.T):
                    y_pred = predict(X)
                    masks = [y_pred == label for label in self.base_classes]
                    for mask, ax in zip(masks, axs_row):
                        plot_decision_boundary(X[mask, :], y_true[mask] if y_true is not None else None, clf_name,
                                               self.feature_names, X_display[mask, :] if X_display is not None else None,
                                               predict=predict, class_names=class_names, zlim=zlim,
                                               idx_x=idx_x, idx_y=idx_y,
                                               fig=fig, ax=ax, **kwargs)
        else:
            if kind == 'proba':
                predict_functions = [self.clf_a.predict_proba, self.clf_b.predict_proba]
                if zlim is None:
                    zlim = (0, 1)
            elif kind == 'log-odds':
                predict_functions = [lambda X: _calc_log_odds_from_log_proba(self.clf_a.predict_log_proba(X)),
                                     lambda X: _calc_log_odds_from_log_proba(self.clf_b.predict_log_proba(X))]
                if zlim is None:
                    zlim = (-4, 4)
                y_true[y_true == self.mclass_diff_clf.base_classes_[0]] = zlim[0]
                y_true[y_true == self.mclass_diff_clf.base_classes_[1]] = zlim[1]
            else:
                raise Exception(f'unsupported kind: {kind}')

            if self.is_binary_classification_task and not separate:
                plot_classes = [1]
            else:
                plot_classes = self.base_classes

            if figsize is None:
                figsize = (2*7, len(plot_classes)*7)
            fig, axs = plt.subplots(nrows=len(plot_classes), ncols=2, sharex=True, sharey=True,
                                    figsize=figsize, constrained_layout=True, squeeze=False)
            for clf_name, predict, axs_row in zip(self.classifier_names, predict_functions, axs.T):
                for class_idx, ax in zip(plot_classes, axs_row.flatten()):
                    predict_class = lambda X: predict(X)[:, class_idx]
                    plot_decision_boundary(X, y_true, clf_name, self.feature_names, X_display,
                                           predict=predict_class, class_names=class_names, zlim=zlim,
                                           idx_x=idx_x, idx_y=idx_y,
                                           fig=fig, ax=ax, **kwargs)

    def check_feature(self, feature):
        return get_index_and_name(self.feature_names, feature)

    def check_class(self, label):
        return get_index_and_name(self.class_names, label)

    def plot_decision_boundaries(self, X, X_display=None, kind='label', x=0, y=1,
                                 xlim=None, ylim=None, zlim=None, type='mclass-diffclf', show=True, **kwargs):
        x_idx, x_name = self.check_feature(x)
        y_idx, y_name = self.check_feature(y)

        if kind == 'label':
            binary_label_diff = self.predict_bin_diff(X)
            label_diff = self.predict_mclass_diff(X)
            if type == 'bin-diffclf':
                z = binary_label_diff
                predict = self.predict_bin_diff
                class_names = self.bin_diff_clf.classes_
            elif type == 'mclass-diffclf':
                z = label_diff
                predict = self.predict_mclass_diff
                class_names = self.class_names
            else:
                raise Exception('invalid type: ' + str(type))

            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
            plot_decision_boundary(X, z,
                                   feature_names=self.feature_names,
                                   X_display=X_display,
                                   predict=predict,
                                   class_names=class_names,
                                   fig=fig, ax=ax,
                                   idx_x=x_idx, idx_y=y_idx,
                                   xlim=xlim, ylim=ylim, zlim=zlim, **kwargs)
        else:
            if kind == 'proba':
                predict_binary = self.predict_bin_diff_proba
                predict_multiclass = self.predict_mclass_diff_proba
                if zlim is None:
                    zlim = 0, 1
            elif kind == 'log-odds':
                predict_binary = self.predict_bin_diff_log_odds
                predict_multiclass = self.predict_mclass_diff_log_odds
                if zlim is None:
                    zlim = -4, 4
            else:
                raise Exception(f'unsupported kind: {kind}')

            binary_diff_predictions = predict_binary(X)
            diff_predictions = predict_multiclass(X)
            outcome_space_readable = 'Probability' if kind == 'proba' else 'Log odds'

            if type == 'bin-diffclf':
                fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
                plot_decision_boundary(X, binary_diff_predictions, feature_names=self.feature_names, predict=predict_binary,
                                       zlim=zlim, fig=fig, ax=ax, xlim=xlim, ylim=ylim,
                                       z_label=outcome_space_readable + ' of different outcomes', **kwargs)
            elif type == 'mclass-diffclf':
                nclasses = len(self.mclass_diff_clf.base_classes_)
                fig, axs = plt.subplots(nrows=nclasses, ncols=nclasses, sharex=True, sharey=True,
                                        figsize=(nclasses*7, nclasses*7), constrained_layout=True)
                for class_idx, ax in zip(self.mclass_diff_clf.classes_, axs.flatten()):
                    class_name = str(self.mclass_diff_clf.class_tuples_[class_idx])
                    predict = lambda X: predict_multiclass(X)[:, class_idx]
                    plot_decision_boundary(X, diff_predictions[:, class_idx], class_name, self.feature_names,
                                           predict=predict, zlim=zlim, fig=fig, ax=ax, xlim=xlim, ylim=ylim,
                                           z_label=f'{outcome_space_readable} of {class_name}', **kwargs)
            else:
                raise Exception('invalid type: ' + str(type))

        if show:
            plt.show()
                
    def plot_confusion_matrix(self, X):
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        cm = confusion_matrix(pred_a, pred_b, labels=self.base_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.base_classes)
        fig, ax = plt.subplots(constrained_layout=True)
        disp.plot(ax=ax)
        ax.set_ylabel('predictions of A')
        ax.set_xlabel('predictions of B')
        plt.show()


def _encode_one_hot(labels, classes):
    indices = np.searchsorted(classes, labels)
    return np.eye(len(classes))[indices]


def _calc_binary_log_odds_from_log_proba(log_proba):
    return log_proba[:, 1] - log_proba[:, 0]


def _calc_log_odds_from_log_proba(log_proba):
    log_odds = np.empty(log_proba.shape)
    for i in range(log_proba.shape[1]):
        class_mask = [True] * log_proba.shape[1]
        class_mask[i] = False
        log_odds[:, i] = log_proba[:, i] - logsumexp(log_proba[:, class_mask], axis=1)
    return log_odds

