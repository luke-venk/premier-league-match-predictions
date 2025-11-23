"""
Configuration parameters for our model.
"""
from enum import Enum

# The number of previous matches to be included to represent a team's current form.
N_MATCHES = 5

# A list of all possible models that our project supports.
class Models(Enum):
    LOGISTIC_REGRESSION = 1
    XGBOOST             = 2
    RANDOM_FOREST       = 3
    SVM                 = 4  
    MLPFFNN             = 5   
    NAIVE_BAYES         = 6
    LIGHTGBM            = 7
    VOTING              = 8

# The specific model we plan to train and evaluate. Options:
MODEL = Models.RANDOM_FOREST

# The year ending the Premier League season we would like to consider.
# e.g., 25 would indicate the Premier League 24/25 season.
END_YEAR = 25
# The number of seasons will determine the beginning season to being
# our dataset. For example, if the ending year was 25, indicating the
# 24/25 season, if we used 10 seasons, the beginning season would be 
# the 15/16 season.
NUM_SEASONS = 10

# The acronym of the betting company whose odds we want to use. Some options are:
# aggregate
# B365
SPORTSBOOK = "aggregate"