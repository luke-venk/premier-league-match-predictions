"""
Class to suppress the warning "UserWarning: X does not have valid feature names,
but LGBMClassifier was fitted with feature names". Issue arises because during
fit, LGBMClassifier receives a Pandas DF, but during predict, the output of the
SimpleImputer is a NumPy array without feature names. This causes issues for
LGBMClassifer but not the other models.
"""
import pandas as pd
from sklearn.impute import SimpleImputer

class DFImputer(SimpleImputer):
    """Returns a Pandas DataFrame instead of a NumPy array."""
    def transform(self, X):
        arr = super().transform(X)
        return pd.DataFrame(arr, columns=X.columns, index=X.index)