import shap
import numpy as np
import xgboost as xgb
from helper_models import LogProbabilityMixin
from sklearn.model_selection import train_test_split


class XGBClassifierWithLogProbaPredict(xgb.XGBClassifier, LogProbabilityMixin):
    pass


hours_per_week_idx = 10
country_idx = 11
capital_loss_idx = 9
race_idx = 6

country_us_value = 0

hyper_parameters = {'nestimators': 100, 'max_depth': 2}
feature_names = np.array(['Age', 'Workclass', 'Education-Num', 'Marital Status',
                          'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain',
                          'Capital Loss', 'Hours per week', 'Country'])
categorical_features = ['Workclass', 'Education-Num', 'Marital Status', 'Occupation',
                        'Relationship', 'Race', 'Sex', 'Country']
feature_precisions = [0 for _ in feature_names]


def apply_influential_modification(X):
    X[:, hours_per_week_idx] -= 2
    return X


def apply_uninfluential_modification(X, seed):
    rng = np.random.default_rng(seed)
    noise = rng.integers(low=0, high=2, size=len(X))
    X[:, capital_loss_idx] += noise
    return X


def make_task_without_modifications():
    X, y = shap.datasets.adult()
    X = X.values
    X_display = shap.datasets.adult(display=True)[0].values
    X_train, X_test, X_display_train, X_display_test, y_train, y_test = \
        train_test_split(X, X_display, y, train_size=0.97, stratify=y, random_state=52)

    clf = XGBClassifierWithLogProbaPredict(**hyper_parameters)
    clf.fit(X_train, y_train)

    return clf, X_test, X_display_test, y_test, feature_names, categorical_features, feature_precisions


def make_task_without_noise():
    X, y = shap.datasets.adult()
    X = X.values
    X_display = shap.datasets.adult(display=True)[0].values
    X_train, X_test, X_display_train, X_display_test, y_train, y_test = \
        train_test_split(X, X_display, y, train_size=0.97, stratify=y, random_state=52)

    clf_a = XGBClassifierWithLogProbaPredict(**hyper_parameters)
    clf_a.fit(X_train, y_train)

    X_train2 = X_train.copy()
    X_train2 = apply_influential_modification(X_train2)
    clf_b = XGBClassifierWithLogProbaPredict(**hyper_parameters)
    clf_b.fit(X_train2, y_train)

    return clf_a, clf_b, X_test, X_display_test, y_test, feature_names, categorical_features, feature_precisions


def make_task_with_noise(seed=43):
    X, y = shap.datasets.adult()
    X = X.values
    X_display = shap.datasets.adult(display=True)[0].values
    X = apply_uninfluential_modification(X, seed)
    X_display = apply_uninfluential_modification(X_display, seed)
    X_train, X_test, X_display_train, X_display_test, y_train, y_test = \
        train_test_split(X, X_display, y, train_size=0.97, stratify=y, random_state=52)

    clf_a = XGBClassifierWithLogProbaPredict(**hyper_parameters)
    clf_a.fit(X_train, y_train)

    X_train2 = X_train.copy()
    X_train2 = apply_influential_modification(X_train2)
    clf_b = XGBClassifierWithLogProbaPredict(**hyper_parameters)
    clf_b.fit(X_train2, y_train)

    return clf_a, clf_b, X_test, X_display_test, y_test, feature_names, categorical_features, feature_precisions
