"""
Reusable utilities for running hyperparameter searches in notebook files.
"""
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from src.config import Models

def tune_model(name, model, params, X_train, y_train, cv=5):
    """
    Goes through the parameter grid configuration for a given model and determines
    its best performing hyperparameters.
    
    Args:
        name: The name of the model.
        model: The model object corresponding to the model.
        params: The hyperparameter configuration to search.
        X_train: The features for the training data.
        y_train: The labels for the training data.
        cv: The number of cross-validation folds.
        
    Returns:
        Dictionary containing the model's best hyperparameters.
    """
    print(f'Tuning {name}...')
    
    start = time.time()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=cv,
        scoring='accuracy',
        n_jobs=4,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed = time.time() - start
    print(f'Finished tuning {name} in {elapsed:.2f} seconds.')
    print(f'Best score = {grid_search.best_score_}.')
    print(f'Best params = {grid_search.best_params_}\n.')
    
    return {
        'name': name,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'grid_search_object': grid_search
    }
    
def search_models(param_grids, X_train, y_train):
    """
    Goes through each of the parameter grid configurations and tunes each
    type of model to determine the best performing hyperparameters for
    each model. param_grids should be defined in the notebook file performing
    the tuning, and should be structured like this:
    
    param_grids = {
        "<name>": {
            "model": <MODEL(...)>,
            "params": {
                "<param1>": ...,
                ...
            }
        },
        ...
    }
    
    Args:
        param_grids: The grid of hyperparameter configurations to search.
        X_train: The features for the training data.
        y_train: The labels for the training data.
        
    Returns:
        List of dictionaries containing each model's best hyperparameters.
    """
    grid_searches = []
    
    for name, cfg in param_grids.items():
        grid_search = tune_model(
            name=name,
            model=cfg['model'],
            params=cfg['params'],
            X_train=X_train,
            y_train=y_train
        )
        grid_searches.append(grid_search)