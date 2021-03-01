from imldiff.model import SKLearnModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class LogisticRegressionModel(SKLearnModel):
    def _make_model(self):
        return LogisticRegression(max_iter=10000)
    
    
class DecisionTreeModel(SKLearnModel):
    def _make_model(self):
        return  DecisionTreeClassifier(max_depth=5)
