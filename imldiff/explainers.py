import shap
import numpy as np
from shap.maskers import Independent
import models


class Explainer:

    def __init__(self, model, expected_value=None, explanation=None):
        self.model = model
        self.expected_value = expected_value
        self.shap_values = explanation

    def generate(self, X, background_data, feature_names, display_data=None):
        pass

    def __str__(self):
        return str(self.model)


class ProbabilityExplainer(Explainer):

    def generate(self, X, background_data, feature_names, display_data=None):
        predict = lambda X: self.model.predict_proba(X)[:, 1]
        self.expected_value = np.mean(predict(X))
        masker = Independent(background_data)
        explainer = shap.Explainer(predict, masker, feature_names=feature_names)
        self.shap_values = explainer(X)
        self.shap_values.display_data = display_data


class LogOddsExplainer(Explainer):

    def generate(self, X, background_data, feature_names, display_data=None):
        predict = lambda X: self.model.predict_log_odds(X)[:, 1]
        self.expected_value = np.mean(predict(X))
        masker = Independent(background_data)
        explainer = shap.Explainer(predict, masker, feature_names=feature_names)
        self.shap_values = explainer(X)
        self.shap_values.display_data = display_data


class DifferenceExplainer:

    def __init__(self, comparer):
        self.proba_c1 = ProbabilityExplainer(comparer.c1)
        self.proba_c2 = ProbabilityExplainer(comparer.c2)
        self.log_odds_c1 = LogOddsExplainer(comparer.c1)
        self.log_odds_c2 = LogOddsExplainer(comparer.c2)
        self.proba_c1_pos_and_c2_neg = ProbabilityExplainer(comparer.c1_pos_and_c2_neg)
        self.proba_c1_neg_and_c2_pos = ProbabilityExplainer(comparer.c1_neg_and_c2_pos)
        self.proba_c1_and_c2_diff = ProbabilityExplainer(comparer.c1_and_c2_diff)
        self.log_odds_c1_pos_and_c2_neg = LogOddsExplainer(comparer.c1_pos_and_c2_neg)
        self.log_odds_c1_neg_and_c2_pos = LogOddsExplainer(comparer.c1_neg_and_c2_pos)
        self.log_odds_c1_and_c2_diff = LogOddsExplainer(comparer.c1_and_c2_diff)
        self.log_odds_ratio = LogOddsExplainer(comparer.log_odds_ratio)

    @property
    def proba_single_model_explainers(self):
        return [self.proba_c1, self.proba_c2]

    @property
    def log_odds_single_model_explainers(self):
        return [self.log_odds_c1, self.log_odds_c2]

    @property
    def single_model_explainers(self):
        return self.proba_single_model_explainers + self.log_odds_single_model_explainers

    @property
    def proba_diff_explainers(self):
        return [self.proba_c1_pos_and_c2_neg, self.proba_c1_neg_and_c2_pos, self.proba_c1_and_c2_diff]

    @property
    def log_odds_diff_explainers(self):
        return [self.log_odds_c1_pos_and_c2_neg, self.log_odds_c1_neg_and_c2_pos, self.log_odds_c1_and_c2_diff, self.log_odds_ratio]

    @property
    def diff_explainers(self):
        return self.proba_diff_explainers + self.log_odds_diff_explainers

    @property
    def explainers(self):
        return self.single_model_explainers + self.diff_explainers

    def generate(self, X, background_data, feature_names, display_data=None):
        for explainer in self.explainers:
            explainer.generate(X, background_data, feature_names, display_data)

