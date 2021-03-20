import os
from datetime import datetime
import pickle
import numpy as np

from imldiff.result import Result


class Model:
    def __repr__(self):
        return self.__class__.__name__

    def predict(self, X):
        return Result(self._prediction_str, self(X))

    @property
    def _prediction_str(self):
        raise NotImplementedError()

    def __call__(self, X):
        raise NotImplementedError()


class TrainableModel(Model):
    def train(self, *args, **kwargs):
        started = datetime.now()
        self._fit(*args, **kwargs)
        print(f'Finished training: {repr(self)} ({datetime.now() - started})')

    def _fit(self, *args, **kwargs):
        raise NotImplementedError()


class ModelLoadException(Exception):
    pass


class SaveableTrainableModel(TrainableModel):
    def __init__(self, state=None):
        super(SaveableTrainableModel, self).__init__(state)
        self._filename = os.path.join('..', 'models', repr(self))

    def load_or_train(self, *args, **kwargs):
        try:
            self._load()
            print(f'Loaded: {repr(self)}')
        except ModelLoadException:
            self.train(*args, **kwargs)

    def _load(self):
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        super(SaveableTrainableModel, self).train(*args, **kwargs)
        self._save()
        print(f'Saved: {repr(self)}')

    def _save(self):
        raise NotImplementedError()


class PickleableTrainableModel(SaveableTrainableModel):
    def _load(self):
        try:
            with open(self._filename, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise ModelLoadException()

    def _save(self):
        with open(self._filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=5)
