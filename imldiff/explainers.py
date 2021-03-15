import os
from datetime import datetime
import pickle
import shap
from shap.maskers import Independent


class ExplainerLoadException(Exception):
    pass


class Explainer:

    group_name = None
    
    def __init__(self, identifier):
        self.identifier = identifier
        if self.group_name:
            self._filename = os.path.join(self._make_path(), self.identifier + '.exp')

    def _make_path(self):
        path = os.path.join('explanations', self.group_name)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        return path

    def load_or_generate(self, model, X, **kwargs):
        try:
            self._load()
            print('Loaded explanation: ' + self._filename)
        except ExplainerLoadException:
            self.generate(model, X, **kwargs)
            
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

    def __str__(self):
        return self.identifier


def make_shap_explainer(identifier, model, X, display_data=None, feature_names=None, max_samples=1000):
    explainer = SHAPExplainer(identifier)
    explainer.load_or_generate(model, X, display_data=display_data, feature_names=feature_names, max_samples=max_samples)
    return explainer


def make_shap_explainer_from_shap_values(identifier, shap_values, data=None, display_data=None):
    explainer = SHAPExplainer(identifier)
    explainer.shap_values = shap_values
    if data:
        explainer.data = data
    if display_data:
        explainer.display_data = display_data
    return explainer


class SHAPExplainer(Explainer):

    def _load(self):
        try:
            with open(self._filename, 'rb') as f:
                self.shap_values = pickle.load(f)
        except FileNotFoundError:
            raise ExplainerLoadException()
        
    def _save(self):
        with open(self._filename, 'wb') as f:
            pickle.dump(self.shap_values, f)

    def _make_explanation(self, model, X, display_data=None, feature_names=None, max_samples=1000):
        self.masker = Independent(X, max_samples=max_samples)
        self.explainer = shap.Explainer(model, self.masker, feature_names=feature_names)
        self.shap_values = self.explainer(X)
        self.shap_values.display_data = display_data
