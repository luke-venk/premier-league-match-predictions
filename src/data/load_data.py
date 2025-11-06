"""
Read and clean the raw CSVs from our data source, and load into a Pandas DataFrame.
"""

import pandas as pd

def load_matches(csv_path: str) -> pd.DataFrame:
    """
    Load our data from the CSVs to a Pandas DataFrame for our model to use.
    
    Could possibly be extended to curl the dataset from online if the raw data
    for the configured season has not already been loaded:
        ex: curl -o data_24_25.csv https://football-data.co.uk/mmz4281/2425/E0.csv    
    
    Args:
        csv_path: The path to the CSV.
    
    Returns:
        A DataFrame with our data.
    """
    df = pd.read_csv(csv_path)
    
    # TODO
    
    return df