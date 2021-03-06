import shap
from shap.maskers import Independent
from imldiff.explainer import BaseSHAPExplainer


class SHAPExplainer(BaseSHAPExplainer):
    def _make_explanation(self, model, X, display_data=None, feature_names=None):
        self.masker = Independent(X, max_samples=100)
        self.explainer = shap.Explainer(model, self.masker, feature_names=feature_names)
        self.shap_values = self.explainer(X)
        self.shap_values.display_data = display_data

