import shap
from shap.maskers import Independent
from imldiff.explainer import BaseSHAPExplainer


class SHAPExplainer(BaseSHAPExplainer):

    def _make_explanation(self, model, X, display_data=None, feature_names=None):
        masker = Independent(X, max_samples=1000)
        explainer = shap.Explainer(model, masker, feature_names=feature_names)
        self.shap_values = explainer(X)
        self.shap_values.display_data = display_data
