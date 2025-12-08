"""
Decoupled from server.py, this module handles the more intense computations
for the inference server, such as loading the model, building the features,
and returning predictions.
"""
from joblib import load
import pandas as pd

from src.config import N_MATCHES, END_YEAR
from src.data.build_features import build_rolling_features, get_feature_columns
from src.util.notebook_utils import get_data_only

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
        self.teams = df['HomeTeam'].unique().tolist()
        self.teams = sorted(self.teams)
        
        # Build the feature DataFrame.
        df_raw = get_data_only()
        self.df = build_rolling_features(df_raw)
        
        # Cache feature columns.
        self.feature_cols = get_feature_columns(self.df.columns)
        
    def _extract_team_stats(self, team: str) -> dict:
        """
        Returns the latest rolling stats and Elo for a team, regardless
        of whether they are home or away.
        """
        # Extract data including the team.
        team_df = self.df[(self.df["home_team"] == team) | (self.df["away_team"] == team)]
        team_df = team_df.sort_values(["season", "date"])
        last = team_df.iloc[-1]
        
        stats = {}
        is_home = (last["home_team"] == team)
        
        # Extract rolling form features, dependent on home or away.
        home_fields = [c for c in self.df.columns if c.endswith("_home")]
        away_fields = [c for c in self.df.columns if c.endswith("_away")]
        
        for h in home_fields:
            base = h.replace("_home", "")
            a = base + "_away"
            
            if is_home and h in team_df.columns:
                stats[base] = last[h]
            elif not is_home and a in team_df.columns:
                stats[base] = last[a]
        
        return stats
    
    def _extract_h2h(self, home_team: str, away_team: str) -> dict:
        df = self.df
        h2h_matches = df[
            (
                ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
                ((df["home_team"] == away_team) & (df["away_team"] == home_team))
            )
        ].sort_values("date").tail(N_MATCHES)
        
        if h2h_matches.empty:
            return {
                "h2h_matches": 0,
                "h2h_draws": 0,
                "h2h_wins_diff": 0,
                "h2h_goals_scored_diff": 0,
                "h2h_goals_conceded_diff": 0,
                "h2h_win_pct_diff": 0.0,
            }
        
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0
        
        for _, row in h2h_matches.iterrows():
            # Determine home team perspective for this match.
            if row["home_team"] == home_team:
                h_goals = row["home_goals"]
                a_goals = row["away_goals"]
                result = row["result"]
            # Determine away team perspective for this match.
            else:
                a_goals = row["home_goals"]
                h_goals = row["away_goals"]
                result = "H" if row["result"] == "A" else ("A" if row["result"] == "H" else "D")
                
            home_goals += h_goals
            away_goals += a_goals
            
            if result == "H":
                home_wins += 1
            elif result == "A":
                away_wins += 1
            else:
                draws += 1
            
        n_matches = len(h2h_matches)
        home_win_pct = home_wins / n_matches
        away_win_pct = away_wins / n_matches
        
        return {
            "h2h_matches": n_matches,
            "h2h_draws": draws,
            "h2h_wins_diff": home_wins - away_wins,
            "h2h_goals_scored_diff": home_goals - away_goals,
            "h2h_goals_conceded_diff": away_goals - home_goals,
            "h2h_win_pct_diff": home_win_pct - away_win_pct
        }
        
    def _compute_betting_odds(self, home_team: str, away_team: str) -> [float, float, float]:
        """
        The model relies on bettings odds, which are available before matches.
        Since the user is predicting a hypothetical match, betting odds will
        not be available, so we will need to impute the value.
        
        The imputation strategy will be to compute a mean of their betting
        odds in previous fixtures where the away team played at the home
        team. If this never happened, then just use the mean of each team's
        odds over the course of the data.
        """
        df = self.df
        # See if this exact matchup (away team playing at home team) has happened
        # a sufficient amount of times. If it has, just take an average of these
        # odds.
        same_matchup = df[(df["home_team"] == home_team) & (df["away_team"] == away_team)]
        if len(same_matchup) > 0:
            odds_home = same_matchup["odds_home_win"].mean()
            odds_draw = same_matchup["odds_draw"].mean()
            odds_away = same_matchup["odds_away_win"].mean()
        else:
            # Otherwise, compute the average odds when the home team is home,
            # and when the away team is away.
            home_df = df[(df["home_team"] == home_team)]
            away_df = df[(df["away_team"] == away_team)]
            
            # Likelihood of the home team winning at home.
            odds_home = home_df["odds_home_win"].mean()
            # Likelihood of the away team winning away.
            odds_away = away_df["odds_away_win"].mean()
            
            # Draws can just be the mean of both teams drawing.
            union_df = df[
                (
                    (df["home_team"] == home_team) |
                    (df["home_team"] == away_team) |
                    (df["away_team"] == home_team) |
                    (df["away_team"] == away_team)
                )
            ]
            odds_draw = union_df["odds_draw"].mean()
            
        return odds_home, odds_draw, odds_away
        

    def _build_feature_row(self, home_team: str, away_team: str) -> pd.DataFrame:
        """
        Based on the home and away teams and the DataFrame, engineer all the
        features required for the model to make a prediction.
        """
        # Use helper function to extract stats for both teams.
        home_stats = self._extract_team_stats(home_team)
        away_stats = self._extract_team_stats(away_team)
        
        # Use helper function to compute imputed betting odds.
        odds_home_win, odds_draw, odds_away_win = self._compute_betting_odds(home_team, away_team)
        
        row = {}
        
        # Engineer required features.
        for feature in self.feature_cols:
            # Home and away difference features.
            if feature.endswith("_diff"):
                base = feature.replace("_diff", "")
                row[feature] = home_stats.get(base, 0) - away_stats.get(base, 0)
                
            # Possession features.
            elif feature == "home_possession_pct":
                row[feature] = home_stats.get("possession_pct", 0)
            elif feature == "away_possession_pct":
                row[feature] = away_stats.get("possession_pct", 0)
            
            # Betting odds.
            elif feature == "odds_home_win":
                row[feature] = odds_home_win
            elif feature == "odds_draw":
                row[feature] = odds_draw
            elif feature == "odds_away_win":
                row[feature] = odds_away_win
        
        # Engineer head to head features.
        h2h_stats = self._extract_h2h(home_team, away_team)
        for k, v in h2h_stats.items():
            if k in self.feature_cols:
                row[k] = v
                
        # Engineer Elo features.
        if "elo_diff_pre" in self.feature_cols:
            # Grab most recent team's Elo from data.
            home_last = self.df[(self.df["home_team"] == home_team) | (self.df["away_team"] == home_team)].iloc[-1]
            away_last = self.df[(self.df["home_team"] == away_team) | (self.df["away_team"] == away_team)].iloc[-1]
            row["elo_diff_pre"] = home_last["elo_diff_pre"] - away_last["elo_diff_pre"]

        # Return row in correct column order.
        return pd.DataFrame([row])[self.feature_cols]
        
    def predict(self, home_team: str, away_team: str) -> [str, float, float, float]:
        """
        Given the two teams for a match, assume that this match occurs
        at the end of the most recent complete season (2024/2025). Construct
        the features relevant for this prediction, and use the best model
        to predict who would win the match.
        
        Args:
            home_team (str): The home team.
            away_team (str): The away team.
        
        Returns:
            prediction (str): Either 'home_win', 'away_win', or 'draw'.
            home_win_probability (float): The probabilitiy of a home team win.
            draw_probability (float): The probabilitiy of a draw.
            away_win_probability (float): The probabilitiy of an away team win.
        """
        # Build synthetic feature row for this hypothetical match.
        feature_row = self._build_feature_row(home_team, away_team)
        
        # Extract prediction probabilities.
        probabilities = self.model.predict_proba(feature_row)[0]
        classes = self.model.classes_
        # The inverse of this label map is found in split.py.
        label_map = {
            0: "home_win",
            1: "draw",
            2: "away_win"
        }
        
        # Determine highest probability outcome.
        best_idx = probabilities.argmax()
        prediction = label_map[classes[best_idx]]
        
        # Output probabilities of each of the 3 classes.
        home_win_probability = float(probabilities[0])
        draw_probability = float(probabilities[1])
        away_win_probability = float(probabilities[2])
        
        return prediction, home_win_probability, draw_probability, away_win_probability
        
    
    def get_teams(self):
        return self.teams