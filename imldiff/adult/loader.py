from shap import datasets
from sklearn.utils import resample


def load_data(n_samples=None):
    X, y = datasets.adult()
    display_data = datasets.adult(display=True)[0].values
    if n_samples is not None:
        return resample(X, y, display_data, n_samples=n_samples, replace=False, stratify=y, random_state=0)
    else:
        return X, y, display_data