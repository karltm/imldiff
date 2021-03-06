import os
from datetime import datetime
import pickle


class ExplainerLoadException(Exception):
    pass


class Explainer:
    
    def __init__(self, identifier):
        self.identifier = identifier
        self._filename = os.path.join('explanations', self.identifier + '.exp')
        
    def load_or_generate(self, model, X):
        try:
            self._load()
            print('Loaded explanation: ' + self._filename)
        except ExplainerLoadException:
            self.generate(model, X)
            
    def _load(self):
        pass
    
    def generate(self, model, X, **kwargs):
        started = datetime.now()
        self._make_explanation(model, X, **kwargs)
        print(f'Finished generating: {self._filename} ({datetime.now() - started})')
        self._save()
        
    def _make_explanation(self, model, X, **kwargs):
        pass
    
    def _save(self):
        pass

    
class BaseSHAPExplainer(Explainer):

    def _load(self):
        try:
            with open(self._filename, 'rb') as f:
                self.shap_values = pickle.load(f)
        except FileNotFoundError:
            raise ExplainerLoadException()
        
    def _save(self):
        with open(self._filename, 'wb') as f:
            pickle.dump(self.shap_values, f)
