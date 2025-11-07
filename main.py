"""
Entry point for our program. This runs the entire pipeline, from loading the data, feature engineering,
training the model, and evaluation.
"""
import os
import pandas as pd
from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, N_MATCHES, SPORTSBOOK
from src.data.load_data import load_matches
from src.data.build_features import build_rolling_features

def main():
    # Debug flag to always redo dataset until proven cleaning works.
    # TODO: remove
    RESET = True
    
    # If the processed data for the configured season has already been loaded, reuse it.
    if os.path.exists(PROCESSED_DATA_PATH) and not RESET:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    # Otherwise, load and process the dataset, saving it to the processed data directory for future use.
    else:
        df_raw = load_matches(csv_path=RAW_DATA_PATH, sportsbook=SPORTSBOOK)
        df = build_rolling_features(df=df_raw, n_matches=N_MATCHES)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(df.head())
        

if __name__ == "__main__":
    main()