"""
Configuration parameters for our model.
"""
from enum import Enum

# The number of previous matches to be included to represent a team's current form.
N_MATCHES = 5

# A list of all possible models that our project supports.
class Models(Enum):
    LOGISTIC_REGRESSION = 1
    XGBOOST            = 2
    RANDOM_FOREST      = 3
    SVM                = 4  
    MLPFFNN            = 5   
    NAIVE_BAYES        = 6
    LIGHTGBM           = 7


# The specific model we plan to train and evaluate.
MODEL = Models.LOGISTIC_REGRESSION
#MODEL = Models.XGBOOST
#MODEL = Models.RANDOM_FOREST
#MODEL = Models.SVM
#MODEL = Models.MLPFFNN
#MODEL = Models.NAIVE_BAYES
#MODEL = Models.LIGHTGBM

# The year beginning the Premier League season we would like to consider.
# e.g., 24 would indicate the Premier League 24/25 season.
SEASON = 24

# File path for our raw dataset.
RAW_DATA_PATH = f"data/raw/data_{SEASON}_{SEASON + 1}.csv"

# File path for our processed dataset.
PROCESSED_DATA_PATH = f"data/processed/data_{SEASON}_{SEASON + 1}.csv"

# The acronym of the betting company whose odds we want to use: bet365.
#SPORTSBOOK = "B365"
SPORTSBOOK = "aggregate"