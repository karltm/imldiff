import os
from datetime import datetime
import pickle
import numpy as np


class ModelLoadException(Exception):
    pass
    

class Classifier:
        
    def __init__(self):
        self._filename = os.path.join('..', 'models', str(self))
        self.model = self._make_model()

    def __str__(self):
        return self.__class__.__name__

    def _make_model(self):
        pass
    
    def load_or_train(self, X, y):
        try:
            self._load()
            print('Loaded model: ' + self._filename)
        except ModelLoadException:
            self.train(X, y)

    def _load(self):
        pass
            
    def train(self, X, y):
        started = datetime.now()
        self._fit(X, y)
        print(f'Finished training: {self._filename} ({datetime.now() - started})')
        self._save()
        
    def _fit(self, X, y):
        pass
    
    def _save(self, ):
        pass
            
    def predict_proba(self, X):
        pass

    def predict_log_odds(self, X):
        log_odds = np.log(self._predict_odds(X))
        return log_odds

    def _predict_odds(self, X):
        probabilities = self.predict_proba(X)
        return probabilities / (1 - probabilities)
    
    
class SKLearnClassifier(Classifier):
    
    def _load(self):
        try:
            with open(self._filename, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise ModelLoadException()
            
    def _fit(self, X, y):
        self.model.fit(X, y)
        
    def _save(self):
        with open(self._filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=5)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    def predict_log_odds(self, X):
        log_odds = self.model.predict_log_proba(X)
        return log_odds[:,1] - log_odds[:,0]
    
    def _make_model(self):
        pass
