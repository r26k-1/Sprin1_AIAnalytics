import numpy as np

class LogTransform:
    def __init__(self, variables):
        self.variables = variables
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        X[self.variables] = X[self.variables].apply(lambda x: np.log1p(x))
        return X
