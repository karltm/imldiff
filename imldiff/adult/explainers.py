import shap
from shap.maskers import Independent
from imldiff.explainer import BaseSHAPExplainer


class SHAPExplainer(BaseSHAPExplainer):
    def _make_explanation(self, model, X):
        masker = Independent(X, max_samples=100)
        explainer = shap.Explainer(model, masker)
        self.shap_values = explainer(X)
        
    def set_display_data(self, display_data):
        self.shap_values.display_data = display_data
