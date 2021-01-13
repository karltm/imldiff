import os
import shap
import sklearn
import xgboost as xgb
import numpy as np
import pickle
from datetime import datetime


class ModelLoadException(Exception):
    pass
    

class Model:
        
    def __init__(self):
        self._filename = os.path.join('models', self._get_name() + '.model')
        self.model = self._make_model()
        
    def _make_model(self):
        pass
    
    def load_or_train(self, X, y):
        try:
            self._load()
            print('Loaded model: ' + self._filename)
        except ModelLoadException:
            self.train(X, y)
            
    def train(self, X, y):
        started = datetime.now()
        self._fit(X, y)
        print(f'Finished training: {self._filename} ({datetime.now() - started})')
        self._save()
        
    def _fit(self, X, y):
        pass
    
    def _save(self, ):
        pass
            
    def predict(self, X):
        pass
    

class LogisticModel(Model):
    
    def _get_name(self):
        return 'Logistic'

    def _make_model(self):
        return sklearn.linear_model.LogisticRegression(max_iter=10000)
        
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
            
    def predict(self, X):
        pass
    

class LogisticProbabilityModel(LogisticModel):

    def predict(self, X):
        return self.model.predict_proba(X)[:,1]
    

class LogisticLogOddsModel(LogisticModel):

    def predict(self, X):
        log_odds = self.model.predict_log_proba(X)
        return log_odds[:,1] - log_odds[:,0]
    
    
class XGBoostModel(Model):
    
    def _get_name(self):
        return 'XGBoost'
        
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
        
    def predict(self, X):
        pass
    
    
class XGBoostProbabilityModel(XGBoostModel):
    
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]
    
    
class XGBoostLogOddsModel(XGBoostProbabilityModel):
    
    def predict(self, X):
        return np.log(self._predict_odds(X))
    
    def _predict_odds(self, X):
        probabilities = super(XGBoostLogOddsModel, self).predict(X)
        return probabilities / (1 - probabilities)
