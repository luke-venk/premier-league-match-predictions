"""
Configuration parameters for our model.
"""
from enum import Enum, auto

# The number of previous matches to be included to represent a team's current form.
N_MATCHES = 5

# A list of all possible models that our project supports.
class Models(Enum):
    LOGISTIC_REGRESSION = auto()
    XGBOOST             = auto()
    RANDOM_FOREST       = auto()
    SVM                 = auto()
    MLPFFNN             = auto()
    NAIVE_BAYES         = auto()
    LIGHTGBM            = auto()
    VOTING              = auto()

# The specific model we plan to train and evaluate. Options:
MODEL = Models.LOGISTIC_REGRESSION

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

# Enable/disable Elo rating features
USE_ELO = True

# Enable/disable head-to-head (H2H) history features
USE_H2H = True

# Enable/disable difference features (home - away)
USE_DIFF = True

# When using difference features, delete the original home/away columns
# This reduces feature dimensionality and can help with some models
DELETE_ORIGINAL_DIFF = True

# Use a curated subset of features (only available when USE_DIFF=True)
# When True, uses only these 11 features:
#   - form_possession_pct_diff, elo_diff_pre, form_goals_conceded_diff
#   - form_win_streak_diff, form_shots_on_target_diff
#   - odds_home_win, odds_away_win, odds_draw
#   - form_goals_scored_diff, form_wins_diff, h2h_draws
FEWER_FEATURES = False