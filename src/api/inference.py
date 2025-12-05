"""
Handles the more intense computations for the inference server, such as
loading the model, building the features, and returning predictions.
"""
from joblib import load
import pandas as pd

from src.config import END_YEAR

class Predictor:
    def __init__(self, model_path: str, end_year: int=END_YEAR):
        # Our analysis in 03_model_performance.ipynb showed that our voting
        # ensemble method performed the best, so deploy this one to the
        # inference server.
        self.model = load(model_path)
        
        # Determine the unique teams from the latest year it was trained on.
        self.teams = []
        raw_csv_path = f'data/raw/raw_{end_year - 1}_{end_year}.csv'
        df = pd.read_csv(raw_csv_path)
        self.teams = df['HomeTeam'].unique()
        
    def predict(self):
        # TODO
        pass
    
    def list_teams(self):
        return self.teams