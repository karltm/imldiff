import pandas as pd
import joblib

categorical_features = ['AGE']
feature_precisions = {'CRIM': 1,
                      'ZN': 0,
                      'INDUS': 2,
                      'NOX': 2,
                      'RM': 2,
                      'AGE': 0,
                      'DIS': 2,
                      'RAD': 2,
                      'TAX': 0,
                      'PTRATIO': 2,
                      'LSTAT': 2}


def make_task3(folder='..'):
    X = pd.read_csv(f'{folder}/test_data.csv')
    y = X['price']
    X = X.drop(columns=['Unnamed: 0', 'price'])
    feature_names = X.columns.to_numpy()
    clf_a = joblib.load(f'{folder}/model_A.joblib')
    clf_b = joblib.load(f'{folder}/model_B.joblib')
    return clf_a, clf_b, X.to_numpy(), y.to_numpy(), feature_names, categorical_features, list(feature_precisions.values())


def make_task4(folder='..'):
    X = pd.read_csv(f'{folder}/test_data.csv')
    y = X['price']
    X = X.drop(columns=['Unnamed: 0', 'price'])
    feature_names = X.columns.to_numpy()
    clf_a = joblib.load(f'{folder}/model_A.joblib')
    clf_b = joblib.load(f'{folder}/model_C.joblib')
    return clf_a, clf_b, X.to_numpy(), y.to_numpy(), feature_names, categorical_features, list(feature_precisions.values())
