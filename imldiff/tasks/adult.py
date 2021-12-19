import shap
import numpy as np
import xgboost as xgb
from helper_models import LogProbabilityMixin
from sklearn.model_selection import train_test_split
from util import index_of
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBClassifierWithLogProbaPredict(xgb.XGBClassifier, LogProbabilityMixin):
    pass


hours_per_week_idx = 10
country_idx = 11

country_germany_value = 11
country_france_value = 10
country_us_value = 0


def modify_hours_per_week(X):
    X[:, hours_per_week_idx] -= 1


def modify_country(X):
    X[:, country_idx] = country_us_value


def make_task_modified_hours_per_week():
    return _make_task(modify_hours_per_week)


def _make_task(modify):
    X, y = shap.datasets.adult()
    feature_names = np.array(X.columns)
    X = X.values
    X_display = shap.datasets.adult(display=True)[0].values
    X_train, X_test, X_display_train, X_display_test, y_train, y_test = \
        train_test_split(X, X_display, y, train_size=0.8, stratify=y, random_state=52)

    clf_a = XGBClassifierWithLogProbaPredict(nestimators=100, max_depth=2)
    clf_a.fit(X_train, y_train)

    X_train2 = X_train.copy()
    modify(X_train2)
    clf_b = XGBClassifierWithLogProbaPredict(nestimators=100, max_depth=2)
    clf_b.fit(X_train2, y_train)

    return clf_a, clf_b, X_test, X_display_test, y_test, feature_names


def make_task_modified_hours_per_week_and_country():
    return _make_task(lambda X: modify_hours_per_week(X) and modify_country(X))





