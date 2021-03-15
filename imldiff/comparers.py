import numpy as np

class Comparer:
    
    def __init__(self, classifier_a, classifier_b):
        self.classifier_a = classifier_a
        self.classifier_b = classifier_b
        
    def predict_proba(self, X):
        predictions_a = self.classifier_a.predict_proba(X)
        predictions_b = self.classifier_b.predict_proba(X)
        return self._merge(predictions_a, predictions_b)
    
    def predict_log_odds(self, X):
        predictions_a = self.classifier_a.predict_log_odds(X)
        predictions_b = self.classifier_b.predict_log_odds(X)
        return self._merge(predictions_a, predictions_b)
    
    def _merge(self, predictions_a, predictions_b):
        pass

    
class DifferenceComparer(Comparer):
    def _merge(self, predictions_a, predictions_b):
        return predictions_a - predictions_b

    def __str__(self):
        return f'Difference of {self.classifier_a} and {self.classifier_b}'
    
    
class AbsoluteDifferenceComparer(DifferenceComparer):
    def _merge(self, predictions_a, predictions_b):
        return abs(super(AbsoluteDifferenceComparer, self)._merge(predictions_a, predictions_b))

    def __str__(self):
        return f'Absolute Difference of {self.classifier_a} and {self.classifier_b}'


class ZeroClippedDifferenceComparer(DifferenceComparer):
    def _merge(self, predictions_a, predictions_b):
        return super(ZeroClippedDifferenceComparer, self)._merge(predictions_a, predictions_b).clip(0)

    def __str__(self):
        return f'Zero-Clipped Difference of {self.classifier_a} and {self.classifier_b}'


class SumComparer(Comparer):
    def _merge(self, predictions_a, predictions_b):
        return predictions_a + predictions_b

    def __str__(self):
        return f'Sum of {self.classifier_a} and {self.classifier_b}'


class SimilarityComparer(Comparer):
    def _merge(self, predictions_a, predictions_b):
        return predictions_a * predictions_b

    def __str__(self):
        return f'Similarity of {self.classifier_a} and {self.classifier_b}'


class DissimilarityComparer(Comparer):
    def _merge(self, predictions_a, predictions_b):
        return predictions_a * (1.0 - predictions_b)

    def __str__(self):
        return f'Dissimilarity of {self.classifier_a} and {self.classifier_b}'


class RatioComparer(Comparer):
    def _merge(self, predictions_a, predictions_b):
        return predictions_a / predictions_b

    def __str__(self):
        return f'Ratio of {self.classifier_a} and {self.classifier_b}'
