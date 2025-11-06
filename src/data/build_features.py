"""
Compute rolling stats to reflect a team's form.
"""

import pandas as pd

def build_rolling_features(df: pd.DataFrame, n_matches: int) -> pd.DataFrame:
    """
    Compute the rolling averages for each team.
    
    Args:
        df: Our raw DataFrame.
    
    Returns:
        A DataFrame with our pre-processed features.
    """
    # TODO
    return df