from sklearn.linear_model import LogisticRegression

class SteppedLogisticRegression(LogisticRegression):
    def decision_function(self, X):
        scores = super(SteppedLogisticRegression, self).decision_function(X)
        return scores.astype(int).astype(float)
