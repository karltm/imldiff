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


def apply_influential_modification(X):
    X[:, hours_per_week_idx] -= 2
    rng = np.random.default_rng(0)
    noise = rng.integers(low=X[:, race_idx].min(), high=X[:, race_idx].max(), size=len(X))
    X[:, race_idx] = noise
    return X


def apply_uninfluential_modification(X):
    rng = np.random.default_rng(1)
    noise = rng.integers(low=X[:, race_idx].min(), high=X[:, race_idx].max(), size=len(X))
    X[:, race_idx] = noise
    return X


def make_task_with_influential_modification():
    return _make_task(apply_influential_modification)


def _make_task(modify):
    X, y = shap.datasets.adult()
    feature_names = np.array(X.columns)
    X = X.values
    X_display = shap.datasets.adult(display=True)[0].values
    X_train, X_test, X_display_train, X_display_test, y_train, y_test = \
        train_test_split(X, X_display, y, train_size=0.97, stratify=y, random_state=52)

    rng = np.random.default_rng(0)
    noise = rng.integers(low=X[:, race_idx].min(), high=X[:, race_idx].max(), size=len(X_train))
    X_train[:, race_idx] = noise

    clf_a = XGBClassifierWithLogProbaPredict(nestimators=100, max_depth=1)
    clf_a.fit(X_train, y_train)

    X_train2 = X_train.copy()
    X_train2 = modify(X_train2)
    clf_b = XGBClassifierWithLogProbaPredict(nestimators=100, max_depth=1)
    clf_b.fit(X_train2, y_train)

    return clf_a, clf_b, X_test, X_display_test, y_test, feature_names


def make_task_with_influential_and_uninfluential_modification():
    return _make_task(lambda X: apply_uninfluential_modification(apply_influential_modification(X)))

