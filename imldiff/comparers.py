from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from difference_models import BinaryDifferenceClassifier, MulticlassDifferenceClassifier
from shap.plots import colors
from scipy.special import logsumexp
import matplotlib.cm as cm



class ModelComparer:
    """ Helper class for comparing two models, that provide API access similar to scikit-learn's models """
    
    def __init__(self, clf_a, clf_b, feature_names):
        self.feature_names = np.array(feature_names)
        self.clf_a = clf_a
        self.clf_b = clf_b
        self.bin_diff_clf = BinaryDifferenceClassifier(clf_a, clf_b)
        self.mclass_diff_clf = MulticlassDifferenceClassifier(clf_a, clf_b)
        
    def fit(self, X, y):
        return self
 
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
        return dict([(clf_name, lambda X, f=f: _encode_one_hot(f(X), self.base_classes))
                     for clf_name, f in self.predict_functions.items()])

    @property
    def predict_proba_functions(self):
        return dict([(clf_name, clf.predict_proba)
                      for clf_name, clf in zip(self.classifier_names, self.classifiers)])

    @property
    def predict_log_odds_functions(self):
        return dict([(clf_name, lambda X, clf=clf: _calc_log_odds_from_log_proba(clf.predict_log_proba(X)))
                     for clf_name, clf in zip(self.classifier_names, self.classifiers)])

    @property
    def predict_bin_diff(self):
        return self.bin_diff_clf.predict
    
    @property
    def predict_bin_diff_proba(self):
        return lambda X: self.bin_diff_clf.predict_proba(X)[:, 1]
    
    @property
    def predict_bin_diff_log_odds(self):
        return lambda X: _calc_binary_log_odds_from_log_proba(self.bin_diff_clf.predict_log_proba(X))
    
    @property
    def predict_mclass_diff(self):
        return self.mclass_diff_clf.predict
    
    @property
    def predict_mclass_diff_one_hot(self):
        return lambda X: _encode_one_hot(self.predict_mclass_diff(X), self.classes)
    
    @property
    def predict_mclass_diff_proba(self):
        return self.mclass_diff_clf.predict_proba
    
    @property
    def predict_mclass_diff_log_odds(self):
        return lambda X: _calc_log_odds_from_log_proba(self.mclass_diff_clf.predict_log_proba(X))

    def plot_individual_clf_decision_boundaries(self, X, X_display=None, y_true=None, separate=False,
                                                kind='label', idx_x=0, idx_y=1, zlim=None, **kwargs):
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
                if zlim is None:
                    zlim = (0, 1)
            elif kind == 'log-odds':
                predict_functions = self.predict_log_odds_functions
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
        
    def plot_decision_boundaries(self, X, X_display=None, kind='label', separate=False, idx_x=0, idx_y=1,
                                 xlim=None, ylim=None, zlim=None, **kwargs):
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
                                       idx_x=idx_x, idx_y=idx_y, xlim=xlim, ylim=ylim, zlim=zlim, **kwargs)
                plot_decision_boundary(X[mask],
                                       label_diff[mask],
                                       'Difference classes for predicted labels',
                                       self.feature_names,
                                       X_display=X_display[mask] if X_display is not None else None,
                                       predict=self.predict_mclass_diff,
                                       class_names=self.class_names,
                                       fig=fig, ax=axs[1],
                                       idx_x=idx_x, idx_y=idx_y, xlim=xlim, ylim=ylim, zlim=zlim, **kwargs)
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
            
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
            fig.suptitle('Binary difference classifier and its decision boundary', fontsize='x-large')
            plot_decision_boundary(X, binary_diff_predictions, 'Labels different', self.feature_names,
                                   predict=predict_binary, zlim=zlim, fig=fig, ax=ax, xlim=xlim, ylim=ylim, **kwargs)
            
            nclasses = len(self.mclass_diff_clf.base_classes_)
            fig, axs = plt.subplots(nrows=nclasses, ncols=nclasses, sharex=True, sharey=True,
                                    figsize=(nclasses*7, nclasses*7), constrained_layout=True)
            fig.suptitle('Multiclass difference classifier and its decision boundaries', fontsize='x-large')
            for class_idx, ax in zip(self.mclass_diff_clf.classes_, axs.flatten()):
                class_name = str(self.mclass_diff_clf.class_tuples_[class_idx])
                predict = lambda X: predict_multiclass(X)[:, class_idx]
                plot_decision_boundary(X, diff_predictions[:, class_idx], class_name, self.feature_names,
                                       predict=predict, zlim=zlim, fig=fig, ax=ax, xlim=xlim, ylim=ylim, **kwargs)
                
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


plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_decision_boundary(X, z=None, title=None, feature_names=None, X_display=None, predict=None,
                           idx_x=0, idx_y=1, class_names=None, zlim=None, mesh_step_size=.5,
                           fig=None, ax=None, xlim=None, ylim=None, predict_value_names=None,
                           show_contour_legend=False, **kwargs):
    """
    - X: instances to plot
    - z: color of instances
    - title: for figure
    - feature_names
    - predict: draw contours for this function (only if number of features is 2)
    - idx_x: index of feature to plot on x axis
    - idx_y: index of feature to plot on y axis
    - class_names: set this to a list of class names if predict returns labels
    - zlim: set this to the range of values if predict returns a continuous variable, e.g. (0, 1)
    - fig, ax
    """


    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if class_names is None and zlim is None:
        if z.dtype == int or z.dtype == bool:
            class_names = np.unique(z)
        else:
            zlim = np.min(z), np.max(z)

    if X_display is None:
        X_display = X

    if z is None:
        z = predict(X)

    if predict_value_names is None:
        predict_value_names = class_names

    draw_contours = predict is not None and X.shape[1] == 2
    if draw_contours:
        if xlim is None:
            xlim = X[:, idx_x].min() - .5, X[:, idx_x].max() + .5
        if ylim is None:
            ylim = X[:, idx_y].min() - .5, X[:, idx_y].max() + .5
        xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], mesh_step_size), np.arange(ylim[0], ylim[1], mesh_step_size))
        z_pred = predict(np.c_[xx.ravel(), yy.ravel()])
        z_pred = z_pred.reshape(xx.shape)

    if class_names is not None:
        legend1 = None
        if draw_contours:
            levels = np.arange(len(predict_value_names) + 1)
            cs = ax.contourf(xx, yy, z_pred + 0.5, levels, colors=plt_colors, alpha=.8)
            proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
                     for pc in cs.collections]
            if show_contour_legend:
                legend1 = ax.legend(proxy, predict_value_names, loc='upper left')
        for class_idx, class_ in enumerate(class_names):
            X_ = X_display[z == class_idx, :]
            if X_.shape[0] == 0:
                continue
            ax.scatter(X_[:, idx_x], X_[:, idx_y], color=plt_colors[class_idx], edgecolors='k', label=str(class_), **kwargs)
        ax.legend()
        if legend1 is not None:
            plt.gca().add_artist(legend1)
    else:
        if draw_contours:
            levels = np.linspace(zlim[0], zlim[1], 21)
            cs = ax.contourf(xx, yy, z_pred, levels, cmap=colors.red_blue, alpha=.8)
            fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.scatter(X[:, idx_x], X[:, idx_y], c=z, cmap=colors.red_blue, vmin=zlim[0], vmax=zlim[1], edgecolors='k', **kwargs)

    if feature_names is not None:
        ax.set_xlabel(feature_names[idx_x])
        ax.set_ylabel(feature_names[idx_y])
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
