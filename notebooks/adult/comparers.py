class Comparer:
    
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        
    def diff_proba(self, X):
        return self.model_b.predict_proba(X) - self.model_a.predict_proba(X)
    
    def diff_abs_proba(self, X):
        return abs(self.model_b.predict_proba(X) - self.model_a.predict_proba(X))
        
    def simil_proba(self, X):
        return self.model_a.predict_proba(X) * self.model_b.predict_proba(X)
        
    # In case you want to multiply those probabilities, they are often getting really small.If you add enough probabilities between > 0 and < 1, you will at some point run intofloating point precision issues. Instead you can use the sum of logarithms. https://www.reddit.com/r/MLQuestions/comments/5lzv9o/sklearn_why_predict_log_proba/
    def simil_logit_sum(self, X):
        return self.model_a.predict_logit(X) + self.model_b.predict_logit(X)
