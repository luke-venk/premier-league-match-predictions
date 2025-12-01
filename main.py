"""
Entry point for our program. This runs the entire pipeline, from loading the data, feature engineering,
training the model, and evaluation.
"""
import os
import pandas as pd
from src.config import END_YEAR, NUM_SEASONS, N_MATCHES, SPORTSBOOK, MODEL, Models
from src.data.load_data import get_processed_path, load_all_seasons
from src.data.build_features import build_rolling_features
from src.data.split import chrono_split
from src.models.train_model import train
from src.models.evaluate_model import evaluate
from src.data.scrape_values import merge_valuations_into_dataframe
from src.data.scrape_possession import merge_possession_into_dataframe

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def main():
    print('>>> Processing data...')
    # Debug flag to always redo dataset until proven cleaning works.
    # TODO: remove
    RESET = True
    
    # If the processed data for the configured season has already been loaded, reuse it.
    PROCESSED_DATA_PATH = get_processed_path(END_YEAR, NUM_SEASONS)
    if os.path.exists(PROCESSED_DATA_PATH) and not RESET:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    # Otherwise, load and process the dataset, saving it to the processed data directory for future use.
    else:
        df_raw = load_all_seasons(end_year=END_YEAR, num_seasons=NUM_SEASONS, sportsbook=SPORTSBOOK)
        df_raw = merge_possession_into_dataframe(df_raw)
        df_raw = merge_valuations_into_dataframe(df_raw, "data/raw/tm_pl_all_columns.csv", "2015-07-01")
        df = build_rolling_features(df=df_raw, n_matches=N_MATCHES)
        df.to_csv(PROCESSED_DATA_PATH, index=False)

        
    # Use a 70-30 chronological train-test split.
    X_train, y_train, X_test, y_test = chrono_split(df, train_ratio=0.7)
    
    # Train the model based on what type of model the user configured.
    print('>>> Training model...')
    model = train(MODEL, X_train, y_train)

    # Evaluate the model based on the holdout set.
    evaluate(model, X_test, y_test, show_confusion_matrix=True)
    
def plot_accuracy_vs_n():
    df_raw = load_all_seasons(end_year=END_YEAR, num_seasons=NUM_SEASONS, sportsbook=SPORTSBOOK)
    df_raw = merge_possession_into_dataframe(df_raw)
    df_raw = merge_valuations_into_dataframe(df_raw, "data/raw/tm_pl_all_columns.csv", "2015-07-01")
    
    # Create several DataFrames using rolling features of different window sizes for
    # computing for statistics.
    n_to_test = np.arange(1, 11)
    
    plt.figure()
    
    for model_type in Models:
        accuracies = []
        for n in n_to_test:
            df = build_rolling_features(df=df_raw, n_matches=n)
            # Use a 70-30 chronological train-test split.
            X_train, y_train, X_test, y_test = chrono_split(df, train_ratio=0.7)
            
            # Train the model based on what type of model the user configured.
            print(f'>>> Training {model_type.name.lower()} with N_MATCHES={n}')
            model = train(model_type, X_train, y_train)
            
            # Evaluate the model based on the holdout set.
            y_pred = model.predict(X_test)
            
            accuracies.append(accuracy_score(y_test, y_pred))
        plt.plot(n_to_test, accuracies, 'o-', label=model_type.name.lower())
        
    plt.xlabel('Number of Games Used to Compute Form Statistics')
    plt.ylabel('Model Accuracy')
    plt.title(f'Accuracy vs. N_MATCHES: NUM_SEASONS={NUM_SEASONS}')
    plt.legend()
    plt.savefig(f'plots/accuracy_vs_n_{NUM_SEASONS}_seasons.png')

if __name__ == "__main__":
    plot_accuracy_vs_n()