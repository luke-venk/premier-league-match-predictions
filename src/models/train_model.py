"""
Train our classifier, depending on the model specified.
"""
import pandas as pd
from src.config import Models
from src.models import logistic_regression
from src.models import XGboost
from src.models import randomforest
from src.models import naivebayes
from src.models import mlpffnn
from src.models import SVM
from src.models import lightGBM
from src.models import voting

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
        model = logistic_regression.train_model(X_train, y_train) #1

    elif model_type == Models.XGBOOST:
        model = XGboost.train_model(X_train, y_train) #2

    elif model_type == Models.RANDOM_FOREST:
        model = randomforest.train_model(X_train, y_train) #3

    elif model_type == Models.SVM:
        model = SVM.train_model(X_train, y_train) #4

    elif model_type == Models.MLPFFNN:
        model = mlpffnn.train_model(X_train, y_train) #5

    elif model_type == Models.NAIVE_BAYES:
        model = naivebayes.train_model(X_train, y_train) #6

    elif model_type == Models.LIGHTGBM:
        model = lightGBM.train_model(X_train, y_train) #7
    
    elif model_type == Models.VOTING:
        model = voting.train_model(X_train, y_train) #8

    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    return model