from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
from imldiff.model import Model, ModelLoadException, SKLearnModel


class LogisticRegressionModel(SKLearnModel):
    def _make_model(self):
        return LogisticRegression(max_iter=10000)
    
    
class XGBoostModel(Model):

    def _make_model(self):
        return xgb.XGBClassifier(nestimators=100, max_depth=2)
        
    def _load(self):
        try:
            self.model.load_model(self._filename)
        except xgb.core.XGBoostError:
            raise ModelLoadException()
        
    def _fit(self, X, y):
        self.model.fit(X, y)
        
    def _save(self):
        self.model.save_model(self._filename)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
