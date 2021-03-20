from imldiff.models import Model, PickleableTrainableModel
from imldiff.result import Result


class Classifier(Model):
    @property
    def _prediction_str(self):
        return f'P({self})'

    def __call__(self, X):
        raise NotImplementedError()

    def is_log_output_space(self):
        return False


class SKLearnClassifier(Classifier, PickleableTrainableModel):
    def _fit(self, X, y):
        self.model.fit(X, y)

    def __call__(self, X):
        return self.model.predict_proba(X)[:, 1]


class MergedClassifier(Classifier):
    def __init__(self, merge, *models):
        super(MergedClassifier, self).__init__()
        self.models = models
        self.merge = merge

    def __call__(self, X):
        return self.predict(X).values

    def predict(self, X):
        results = [model.predict(X) for model in self.models]
        return self.merge(*results)

    def is_log_output_space(self):
        return self.merge.__name__.startswith('calculate_log_of_')

    def __str__(self):
        name = self.merge.__name__
        name = name.removeprefix('calculate_probability_of_')
        name = name.removeprefix('calculate_log_of_')
        name = name.replace('_', ' ')
        name = name.capitalize()
        return name

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"merge={self.merge.__name__}," \
               f"models=({','.join([repr(model) for model in self.models])}))"
