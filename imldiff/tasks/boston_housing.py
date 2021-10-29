import pandas as pd
import joblib


def make_task3(folder='..'):
    X = pd.read_csv(f'{folder}/test_data.csv')
    y = X['price']
    X = X.drop(columns=['Unnamed: 0', 'price'])
    feature_names = X.columns.to_numpy()
    clf_a = joblib.load(f'{folder}/model_A.joblib')
    clf_b = joblib.load(f'{folder}/model_B.joblib')
    return clf_a, clf_b, X, y, feature_names


def make_task4(folder='..'):
    X = pd.read_csv(f'{folder}/test_data.csv')
    y = X['price']
    X = X.drop(columns=['Unnamed: 0', 'price'])
    feature_names = X.columns.to_numpy()
    clf_a = joblib.load(f'{folder}/model_A.joblib')
    clf_b = joblib.load(f'{folder}/model_C.joblib')
    return clf_a, clf_b, X, y, feature_names
