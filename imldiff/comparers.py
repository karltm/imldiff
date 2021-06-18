from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from util import encode_one_hot, calc_binary_log_odds_from_log_proba, calc_log_odds_from_log_proba
from difference_models import BinaryDifferenceClassifier, MulticlassDifferenceClassifier
from plots import plot_decision_boundary

class ModelComparer:
    
    def __init__(self, clf_a, clf_b, feature_names):
        self.feature_names = np.array(feature_names)
        self.clf_a = clf_a
        self.clf_b = clf_b
        self.bin_diff_clf = BinaryDifferenceClassifier(clf_a, clf_b)
        self.mclass_diff_clf = MulticlassDifferenceClassifier(clf_a, clf_b)
        
    def fit(self, X, y):
        self.bin_diff_clf.fit(X, y)
        self.mclass_diff_clf.fit(X, y)
 
    @property
    def classifiers(self):
        return self.clf_a, self.clf_b

    @property
    def classifier_names(self):
        return ['A', 'B']
    
    @property
    def base_classes(self):
        return self.mclass_diff_clf.base_classes_

    @property
    def base_class_names(self):
        return np.array([str(label) for label in self.base_classes])
    
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
    
    @property
    def predict_functions(self):
        return dict([(clf_name, clf.predict)
                     for clf_name, clf in zip(self.classifier_names, self.classifiers)])
    
    @property
    def predict_one_hot_functions(self):
        return dict([(clf_name, lambda X, f=f: encode_one_hot(f(X), self.base_classes))
                     for clf_name, f in self.predict_functions.items()])

    @property
    def predict_proba_functions(self):
        return dict([(clf_name, clf.predict_proba)
                      for clf_name, clf in zip(self.classifier_names, self.classifiers)])

    @property
    def predict_log_odds_functions(self):
        return dict([(clf_name, lambda X, clf=clf: calc_log_odds_from_log_proba(clf.predict_log_proba(X)))
                     for clf_name, clf in zip(self.classifier_names, self.classifiers)])

    @property
    def predict_bin_diff(self):
        return self.bin_diff_clf.predict
    
    @property
    def predict_bin_diff_proba(self):
        return lambda X: self.bin_diff_clf.predict_proba(X)[:, 1]
    
    @property
    def predict_bin_diff_log_odds(self):
        return lambda X: calc_binary_log_odds_from_log_proba(self.bin_diff_clf.predict_log_proba(X))
    
    @property
    def predict_mclass_diff(self):
        return self.mclass_diff_clf.predict
    
    @property
    def predict_mclass_diff_one_hot(self):
        return lambda X: encode_one_hot(self.predict_mclass_diff(X), self.classes)
    
    @property
    def predict_mclass_diff_proba(self):
        return self.mclass_diff_clf.predict_proba
    
    @property
    def predict_mclass_diff_log_odds(self):
        return lambda X: calc_log_odds_from_log_proba(self.mclass_diff_clf.predict_log_proba(X))

    def plot_individual_clf_decision_boundaries(self, X, X_display=None, y_true=None, separate=False,
                                                kind='label', idx_x=0, idx_y=1, **kwargs):
        zlim = None
        class_names = None
        if kind == 'label':
            class_names = self.mclass_diff_clf.base_classes_
            if not separate:
                fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(2*7, 7), constrained_layout=True)
                for (clf_name, predict), ax in zip(self.predict_functions.items(), axs):
                    plot_decision_boundary(X, y_true, clf_name, self.feature_names, X_display,
                                           predict=predict, class_names=class_names, zlim=zlim,
                                           idx_x=idx_x, idx_y=idx_y,
                                           fig=fig, ax=ax, **kwargs)
            else:
                fig, axs = plt.subplots(nrows=len(self.base_classes), ncols=2, sharex=True, sharey=True,
                                        figsize=(2*7, len(self.base_classes)*7),
                                        constrained_layout=True, squeeze=False)
                
                for (clf_name, predict), axs_row in zip(self.predict_functions.items(), axs.T):
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
                predict_functions = self.predict_proba_functions
                zlim = (0, 1)
            elif kind == 'log-odds':
                predict_functions = self.predict_log_odds_functions
                zlim = (-4, 4)
                y_true[y_true == self.mclass_diff_clf.base_classes_[0]] = zlim[0]
                y_true[y_true == self.mclass_diff_clf.base_classes_[1]] = zlim[1]
            else:
                raise Exception(f'unsupported kind: {kind}')

            if self.is_binary_classification_task and not separate:
                plot_classes = [1]
            else:
                plot_classes = self.base_classes

            fig, axs = plt.subplots(nrows=len(plot_classes), ncols=2, sharex=True, sharey=True,
                                    figsize=(2*7, len(plot_classes)*7), constrained_layout=True, squeeze=False)
            for (clf_name, predict), axs_row in zip(predict_functions.items(), axs.T):
                for class_idx, ax in zip(plot_classes, axs_row.flatten()):
                    predict_class = lambda X: predict(X)[:, class_idx]
                    plot_decision_boundary(X, y_true, clf_name, self.feature_names, X_display,
                                           predict=predict_class, class_names=class_names, zlim=zlim,
                                           idx_x=idx_x, idx_y=idx_y,
                                           fig=fig, ax=ax, **kwargs)
        fig.suptitle('Base classification task with decision boundaries of the classifiers', fontsize='x-large')
        
    def plot_decision_boundaries(self, X, X_display=None, kind='label', separate=False, idx_x=0, idx_y=1, **kwargs):
        if kind == 'label':
            binary_label_diff = self.predict_bin_diff(X)
            label_diff = self.predict_mclass_diff(X)
            if separate:
                masks = [binary_label_diff, ~binary_label_diff]
            else:
                masks = [np.repeat(True, len(binary_label_diff))]
                
            fig, axs = plt.subplots(ncols=2, nrows=len(masks),
                                    sharex=True, sharey=True,
                                    figsize=(2*7, len(masks)*7),
                                    constrained_layout=True, squeeze=False)
            if separate:
                axs = axs.T
            fig.suptitle('Difference classifiers and their decision boundaries', fontsize='x-large')
            for mask, axs in zip(masks, axs):
                plot_decision_boundary(X[mask],
                                       binary_label_diff[mask],
                                       'Labels different',
                                       self.feature_names,
                                       X_display=X_display[mask] if X_display is not None else None,
                                       predict=self.predict_bin_diff,
                                       class_names=self.bin_diff_clf.classes_,
                                       fig=fig, ax=axs[0],
                                       idx_x=idx_x, idx_y=idx_y, **kwargs)
                plot_decision_boundary(X[mask],
                                       label_diff[mask],
                                       'Difference classes for predicted labels',
                                       self.feature_names,
                                       X_display=X_display[mask] if X_display is not None else None,
                                       predict=self.predict_mclass_diff,
                                       class_names=self.class_names,
                                       fig=fig, ax=axs[1],
                                       idx_x=idx_x, idx_y=idx_y, **kwargs)
        else:
            if kind == 'proba':
                predict_binary = self.predict_bin_diff_proba
                predict_multiclass = self.predict_mclass_diff_proba
                zlim = 0, 1
            elif kind == 'log-odds':
                predict_binary = self.predict_bin_diff_log_odds
                predict_multiclass = self.predict_mclass_diff_log_odds
                zlim = -4, 4
            else:
                raise Exception(f'unsupported kind: {kind}')
            binary_diff_predictions = predict_binary(X)
            diff_predictions = predict_multiclass(X)
            
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
            fig.suptitle('Binary difference classifier and its decision boundary', fontsize='x-large')
            plot_decision_boundary(X, binary_diff_predictions, 'Labels different', self.feature_names,
                                   predict=predict_binary, zlim=zlim, fig=fig, ax=ax, **kwargs)
            
            nclasses = len(self.mclass_diff_clf.base_classes_)
            fig, axs = plt.subplots(nrows=nclasses, ncols=nclasses, sharex=True, sharey=True,
                                    figsize=(nclasses*7, nclasses*7), constrained_layout=True)
            fig.suptitle('Multiclass difference classifier and its decision boundaries', fontsize='x-large')
            for class_idx, ax in zip(self.mclass_diff_clf.classes_, axs.flatten()):
                class_name = str(self.mclass_diff_clf.class_tuples_[class_idx])
                predict = lambda X: predict_multiclass(X)[:, class_idx]
                plot_decision_boundary(X, diff_predictions[:, class_idx], class_name, self.feature_names,
                                       predict=predict, zlim=zlim, fig=fig, ax=ax, **kwargs)
                
    def plot_confusion_matrix(self, X):
        pred_a = self.clf_a.predict(X)
        pred_b = self.clf_b.predict(X)
        cm = confusion_matrix(pred_a, pred_b, labels=self.base_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.base_classes)
        fig, ax = plt.subplots(constrained_layout=True)
        disp.plot(ax=ax)
        ax.set_title('Confusion matrix of predicted labels of $c_A$ vs. $c_B$')
        ax.set_ylabel('$\hat{y}_A$')
        ax.set_xlabel('$\hat{y}_B$')
        plt.show()
