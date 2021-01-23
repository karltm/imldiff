class Comparer:
    
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b

    
class DifferenceComparer(Comparer):
    
    def predict(self, X):
        return self.model_b.predict(X) - self.model_a.predict(X)
    
    
class AbsoluteDifferenceComparer(DifferenceComparer):
    
    def predict(self, X):
        return abs(super(self, AbsoluteDifferenceComparer).predict(X))
    
    
class SimilarityComparer(Comparer):
    
    def predict(self, X):
        return self.model_a.predict(X) * self.model_b.predict(X)
    
    
class AdditionComparer(Comparer):
    
    def predict(self, X):
        return self.model_a.predict(X) + self.model_b.predict(X)
    