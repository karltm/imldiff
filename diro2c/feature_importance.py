from sklearn import linear_model

from yellowbrick.datasets import load_occupancy
from yellowbrick.model_selection import FeatureImportances


def main():
    # Load the classification data set
    X, y = load_occupancy()

    model = linear_model.RidgeClassifier()
    viz = FeatureImportances(model, relative=False)
    viz.fit(X, y)
    viz.show()


if __name__ == "__main__":
    main()
