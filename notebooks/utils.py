"""
Helper functions for exploratory data analysis to be done in
the notebooks/ directory.
"""
import os
import pandas as pd

from src.data.load_data import get_processed_path, load_all_seasons
from src.config import END_YEAR, NUM_SEASONS, N_MATCHES, SPORTSBOOK, MODEL
from src.data.scrape_possession import merge_possession_into_dataframe
from src.data.scrape_values import merge_valuations_into_dataframe
from src.data.build_features import build_rolling_features
from src.data.split import chrono_split
from src.models.train_model import train
from src.models.evaluate_model import evaluate

def get_data(end_year: int=END_YEAR,
             num_seasons: int=NUM_SEASONS,
             sportsbook: str=SPORTSBOOK,
             n_matches: int=N_MATCHES):
    """
    Helper function to automate X_train, y_train, X_test, and y_test based on
    configuration parameters.
    """
    processed_data_path = get_processed_path(end_year, num_seasons)
    
    # If the processed data for the configured season has already been loaded, reuse it.
    if os.path.exists(processed_data_path):
        df_raw = pd.read_csv(processed_data_path)
    # Otherwise, load and process the dataset, saving it to the processed data directory for future use.
    else:
        # Load data from Football-Data.co.uk aggregated from configured seasons.
        df_raw = load_all_seasons(end_year=end_year, num_seasons=num_seasons, sportsbook=sportsbook)
        # Merge possession data scraped from FootballCritic.
        df_raw = merge_possession_into_dataframe(df_raw)
        # Merge squad valuation data from TransferMarkt.
        df_raw = merge_valuations_into_dataframe(df_raw, "data/raw/tm_pl_all_columns.csv", "2015-07-01")
        # Write to file, so we can reuse as needed.
        df_raw.to_csv(processed_data_path, index=False)
    
    # Engineer feature matrix.    
    df = build_rolling_features(df=df_raw, n_matches=n_matches)
    
    # Use a 70-30 chronological train-test split.
    # returns X_train, y_train, X_test, y_test.
    return chrono_split(df, train_ratio=0.7)

def get_model_accuracy(model: int=MODEL,
                       end_year: int=END_YEAR,
                       num_seasons: int=NUM_SEASONS,
                       sportsbook: str=SPORTSBOOK,
                       n_matches: int=N_MATCHES):
    """
    Prints classification report
    """
    X_train, y_train, X_test, y_test = get_data(end_year, num_seasons, sportsbook, n_matches)
    
    # Train the model based on what type of model the user configured.
    print('>>> Training model...')
    model = train(model, X_train, y_train)

    # Evaluate the model based on the holdout set.
    evaluate(model, X_test, y_test, show_confusion_matrix=True)