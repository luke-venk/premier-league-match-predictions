"""
Train our classifier, depending on the model specified.
"""
import pandas as pd
from src.config import Models
from src.models import logistic_regression

def train(model_type: int, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Regardless of the type of model, train the model.
    
    Args:
        df: The dataframe.
        model_type: The specific type of model we plan to use.
    
    Returns:
        A DataFrame with our data.
    """
    if model_type == Models.LOGISTIC_REGRESSION:
        model = logistic_regression.train_model(X_train, y_train)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    return model