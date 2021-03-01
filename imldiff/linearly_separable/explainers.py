import shap
from shap.maskers import Independent
from imldiff.explainer import BaseSHAPExplainer


class SHAPExplainer(BaseSHAPExplainer):

    def _make_explanation(self, model, X):
        masker = Independent(X, max_samples=1000)
        explainer = shap.Explainer(model, masker)
        self.shap_values = explainer(X)

    @property
    def feature_names(self):
        return self.shap_values.feature_names
    @feature_names.setter
    def feature_names(self, new_feature_names):
        self.shap_values.feature_names = new_feature_names
