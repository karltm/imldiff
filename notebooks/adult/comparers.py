import shap
import sklearn
import xgboost as xgb
import numpy as np
import pickle
import os
from datetime import datetime

X, y = shap.datasets.adult()
display_data = shap.datasets.adult(display=True)[0].values

def make_path(filename):
    return os.path.join('models', filename)


class Model:
    def train(self, X, y):
        pass
    def load_or_train(self, X, y):
        pass
    def predict_proba(self, x):
        pass
    def predict_odds(self, x):
        pass
    def predict_logit(self, x):
        pass
    

class LogisticRegressionModel(Model):
    '''a simple linear logistic model'''
    
    _filename = make_path(__qualname__ + '.pickle')

    def __init__(self):
        self.model = None
        
    def train(self, X, y):
        started = datetime.now()
        self.model = sklearn.linear_model.LogisticRegression(max_iter=10000)
        self.model.fit(X, y)
        print(f'Finished training: {self._filename} ({datetime.now() - started})')
        with open(self._filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=5)
        
    def load_or_train(self, X, y):
        try:
            with open(self._filename, 'rb') as f:
                self.model = pickle.load(f)
                print('Loaded model: ' + self._filename)
        except FileNotFoundError:
            self.train(X, y)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:,1]

    def predict_odds(self, x):
        probabilities = self.predict_proba(x)
        return probabilities / (1 - probabilities)

    def predict_logit(self, x):
        p = self.model.predict_log_proba(x)
        return p[:,1] - p[:,0]
    
    
class XGBModel(Model):
    '''a boosted tree model'''
    
    _filename = make_path(__qualname__ + '.json')
    
    def __init__(self):
        self.model = xgb.XGBClassifier(nestimators=100, max_depth=2)
        
    def train(self, X, y):
        started = datetime.now()
        self.model.fit(X, y)
        print(f'Finished training: {self._filename} ({datetime.now() - started})')
        self.model.save_model(self._filename)
        
    def load_or_train(self, X, y):
        try:
            self.model.load_model(self._filename)
            print('Loaded model: ' + self._filename)
        except:
            self.train(X, y)
    
    def predict_proba(self, x):
        return self.model.predict_proba(x)[:,1]
    
    def predict_odds(self, x):
        probabilities = self.predict_proba(x)
        return probabilities / (1 - probabilities)
    
    def predict_logit(self, x):
        return np.log(self.predict_odds(x))
    

class Comparer:
    def train(self):
        pass
    def load_or_train(self):
        pass
    
    
class LogisticRegressionVsXGBComparer(Comparer):
    
    def __init__(self):
        self.model_a = LogisticRegressionModel()
        self.model_b = XGBModel()
    
    def train(self):
        self.model_a.train(X, y)
        self.model_b.train(X, y)
        
    def load_or_train(self):
        self.model_a.load_or_train(X, y)
        self.model_b.load_or_train(X, y)
        
    def simil_proba(X):
        return self.model_a.predict_proba(X) * self.model_b.predict_proba(X)
        
    # In case you want to multiply those probabilities, they are often getting really small.If you add enough probabilities between > 0 and < 1, you will at some point run intofloating point precision issues. Instead you can use the sum of logarithms. https://www.reddit.com/r/MLQuestions/comments/5lzv9o/sklearn_why_predict_log_proba/
    def simil_logit_sum(X):
        return self.model_a.predict_logit(X) + self.model_b.predict_logit(X)
