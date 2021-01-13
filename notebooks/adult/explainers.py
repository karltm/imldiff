import os
import pickle
from datetime import datetime
import shap

class ExplainerLoadException(Exception):
    pass


class Explainer:
    
    def __init__(self, model, identifier):
        self._model = model
        self._filename = os.path.join('models', identifier + '.exp')
        
    def load_or_generate(self, X):
        try:
            self._load()
            print('Loaded explanation: ' + self._filename)
        except ExplainerLoadException:
            self.generate(X)
            
    def _load(self):
        pass
    
    def generate(self, X):
        started = datetime.now()
        self._make_explanation(X)
        print(f'Finished generating: {self._filename} ({datetime.now() - started})')
        self._save()
        
    def _make_explanation(self, X):
        pass
    
    def _save(self):
        pass
    
    
class SHAPExplainer(Explainer):

    def _load(self):
        try:
            with open(self._filename, 'rb') as f:
                self.shap_values = pickle.load(f)
        except FileNotFoundError:
            raise ExplainerLoadException()

    def _make_explanation(self, X):
        masker = shap.maskers.Independent(X, max_samples=100)
        explainer = shap.Explainer(self._model.predict, masker)
        self.shap_values = explainer(X)
        
    def _save(self):
        with open(self._filename, 'wb') as f:
            pickle.dump(self.shap_values, f)