import numpy as np


number_of_samples = 1000
number_of_features = 2


def load_data():
    rng = np.random.RandomState(2)
    X = rng.uniform(-1, 1, (number_of_samples, number_of_features))
    return X
