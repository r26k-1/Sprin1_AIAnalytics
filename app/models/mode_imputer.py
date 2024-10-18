class ModeImputer:
    def __init__(self, variables):
        self.variables = variables
    
    def fit(self, X):
        self.imputer_values_ = X[self.variables].mode().iloc[0]
        return self
    
    def transform(self, X):
        X[self.variables] = X[self.variables].fillna(self.imputer_values_)
        return X
