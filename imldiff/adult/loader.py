from shap import datasets
from sklearn.utils import resample

def load_data():
    X, y = datasets.adult()
    display_data = datasets.adult(display=True)[0].values
    return X, y, display_data

def load_data_sample(n_samples=1000):
    X, y, display_data = load_data()
    return resample(X, y, display_data, n_samples=n_samples, replace=False, stratify=y, random_state=0)
