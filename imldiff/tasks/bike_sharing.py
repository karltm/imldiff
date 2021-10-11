import pandas as pd
import numpy as np
from datetime import datetime
import xgboost
from sklearn.model_selection import StratifiedKFold, train_test_split
from helper_models import LogProbabilityMixin


class XGBClassifierWithLogProbaPredict(xgboost.XGBClassifier, LogProbabilityMixin):
    pass


def _make_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    skf = StratifiedKFold(n_splits=10)
    best_score = 0
    best_clf = None
    for train_index, test_index in skf.split(X_train, y_train):
        clf = XGBClassifierWithLogProbaPredict(n_jobs=4, use_label_encoder=False, eval_metric='logloss')
        clf.fit(X.iloc[train_index], y.iloc[train_index])
        score = clf.score(X.iloc[test_index], y.iloc[test_index])
        if score > best_score:
            best_clf, best_score = clf, score
    return best_clf


def make_task(file='day.csv', threshold=4500):
    df = pd.read_csv(file, parse_dates=['dteday'])
    is_over_threshold = df['cnt'] > threshold
    is_part1 = df['dteday'] < datetime(2012, 1, 1)

    feature_names = np.array(['season', 'mnth', 'holiday', 'weekday', 'workingday',
                              'weathersit', 'temp', 'atemp', 'hum', 'windspeed'])
    X = df.loc[:, feature_names]
    y = is_over_threshold.astype(int)

    X_train1 = X[is_part1]
    y_train1 = y[is_part1]
    X_train2 = X[~is_part1]
    y_train2 = y[~is_part1]

    clf_a = _make_classifier(X_train1, y_train1)
    clf_b = _make_classifier(X_train2, y_train2)

    return clf_a, clf_b, X_train2.to_numpy(), y_train2.to_numpy(), feature_names

