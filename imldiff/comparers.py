class Comparer:
    
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        
    def predict_proba(self, X):
        return self._predict(self.model_a.predict_proba, self.model_b.predict_proba, X)
    
    def predict_log_odds(self, X):
        return self._predict(self.model_a.predict_log_odds, self.model_b.predict_log_odds, X)
    
    def _predict(self, predict_a, predict_b, X):
        pass

    
class DifferenceComparer(Comparer):
    def _predict(self, predict_a, predict_b, X):
        return predict_b(X) - predict_a(X)

    def __str__(self):
        return f'Difference of {self.model_b} and {self.model_a}'
    
    
class AbsoluteDifferenceComparer(DifferenceComparer):
    def _predict(self, predict_a, predict_b, X):
        return abs(super(AbsoluteDifferenceComparer, self)._predict(predict_a, predict_b, X))

    def __str__(self):
        return f'Absolute Difference of {self.model_b} and {self.model_a}'
    
    
class SimilarityComparer(Comparer):
    def _predict(self, predict_a, predict_b, X):
        return predict_a(X) * predict_b(X)

    def __str__(self):
        return f'Similarity of {self.model_a} and {self.model_b}'
    
    
class SumComparer(Comparer):
    def _predict(self, predict_a, predict_b, X):
        return predict_a(X) + predict_b(X)

    def __str__(self):
        return f'Sum of {self.model_a} and {self.model_b}'
