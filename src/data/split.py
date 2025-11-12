"""
Class to help with chronological 70-30 train-test split. This temporal split
is necessary as opposed to sklearn's train_test_split(), since we are trying
to retain chronological information related to form, as opposed to random
sampling.
"""
import pandas as pd

def chrono_split(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple:
    """
    Performs chronological train-test split and returns
    (X_train, y_train, X_test, y_test).
    
    Args:
        df: The dataframe.
        train_ratio: The percentage of the dataset to reserve for training.
    
    Returns:
        X_train: The features to train the dataset with.
        y_train: The labels associated with the training data.
        X_test: The features to test the dataset with.
        y_test: The labels associated with the test data.
    """
    # Map from results to softmax output.
    label_map = {'H': 0, 'D': 1, 'A': 2}

    # Copy DF to save original.
    df_proc = df.copy()
    # Sort by date again just in case.
    df_proc = df_proc.sort_values('date').reset_index(drop=True)
    
    # Use only previous data and odds as features in the model.
    feature_cols = [c for c in df_proc.columns if c.startswith('form_')]
    feature_cols += ['odds_home_win', 'odds_draw', 'odds_away_win']

    # Make feature matrix X and target y.
    X = df_proc[feature_cols].copy()
    y = df_proc['result'].map(label_map).astype(int)

    # Split the data.
    cut = int(train_ratio * len(df_proc))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y[:cut], y[cut:]
    
    return X_train, y_train, X_test, y_test