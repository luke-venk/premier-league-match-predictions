"""
Compute rolling stats to reflect a team's form.
"""
import numpy as np
import pandas as pd

def build_rolling_features(df: pd.DataFrame, n_matches: int) -> pd.DataFrame:
    """
    Compute the rolling stats of the following features both home and away teams:
        - Wins (last 5 games)
        - Points (last 5 games)
        - Goals scored (last 5 games)
        - Goals conceded (last 5 games)
        - Shots on target (last 5 games)
        - Fouls committed (last 5 games)
        - Win streak length
        - Bookmaker odds (for this match)
    
    Args:
        df: Our raw DataFrame.
    
    Returns:
        A DataFrame with our pre-processed features.
    """
    # Ensure data is chronologically sorted (although, it already should be).
    # Adding 'season' as a grouping key prevents rolling windows from crossing
    # season boundaries.
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    # Add ELo: Season regress starts off new seasons by returning elo closer to base.
    df = add_elo_features(df, K=24.0, base=1500.0, home_adv=60.0, season_regress=0.25)
    
    def compute_team_form(team_name: str) -> pd.DataFrame:
        """Compute rolling form features for a single team."""
        # Only consider entries that include the team as either the home or away team.
        team_df = df[(df["home_team"] == team_name) | (df["away_team"] == team_name)].copy()
        team_df = team_df.sort_values(["season", "date"]).reset_index(drop=True)
        
        # Boolean mask to know whether to use "home" or "away" columns in dataset.
        mask_home = team_df["home_team"] == team_name
        
        # Wins (0 or 1)
        team_df["wins"] = (
            # This team was home and the home team won, OR
            (mask_home & (team_df["result"] == "H")) |
            # This team was away and the away team won.
            (~mask_home & (team_df["result"] == "A"))
        ).astype(int)

        # Points add 3 for a win 1 for a draw and 0 for a loss 
        team_df["points"] = (3 * team_df["wins"] + (team_df["result"] == "D").astype(int)).astype(int)
        
        # Goals scored
        team_df["goals_scored"] = team_df["home_goals"].where(mask_home, team_df["away_goals"])
        
        # Goals conceded
        team_df["goals_conceded"] = team_df["away_goals"].where(mask_home, team_df["home_goals"])
        
        # Shots on target
        team_df["shots_on_target"] = team_df["home_shots_on_target"].where(mask_home, team_df["away_shots_on_target"])
        
        # Fouls committed
        team_df["fouls_committed"] = team_df["home_fouls"].where(mask_home, team_df["away_fouls"])

        # Win streak
        team_df["form_win_streak"] = consecutive_win_streak_before(team_df["wins"])
        
        # For each of the metrics we just computed, calculate the total metrics over the
        # previous n_matches games.
        for col in ["wins", "points", "goals_scored", "goals_conceded", "shots_on_target", "fouls_committed"]:
            # shift() prevents data leakage by shifting the current row down and only including prior rows.
            # rolling(n_matches) creates rolling window of n_matches entries
            team_df[f"form_{col}"] = team_df[col].shift().rolling(n_matches, min_periods=1).sum()
            
        # Add the team name as an identifier.
        team_df["team"] = team_name
        
        # Return only the features useful for the model.
        return team_df[
            [
                "season",
                "date",
                "team",
                "form_wins",
                "form_points",
                "form_goals_scored",
                "form_goals_conceded",
                "form_shots_on_target",
                "form_fouls_committed",
                "form_win_streak"
            ]
        ]
    
    # Call helper function to compute team form for each team in the dataset.
    all_teams = pd.concat(
        [compute_team_form(team) for team in pd.concat([df["home_team"], df["away_team"]]).unique()],
        ignore_index=True
    )
    
    # For each match, attach the home team's form stats (from the all_teams df)
    # based on date and team name.
    df = df.merge(
        all_teams,
        # Merge if df["date"] == all_teams["date"] and df["home_team"] == all_teams["team"]
        left_on=["season", "date", "home_team"],
        right_on=["season", "date", "team"],
        # Keep rows from left (df), and bring in matching rows from right (all_teams)
        how="left",
        # Add "_home" to overlapping column names to distinguish them before the away merge
        suffixes=("", "_home")
    )
    
    # Do the same for the away team.
    df = df.merge(
        all_teams,
        left_on=["season", "date", "away_team"],
        right_on=["season", "date", "team"],
        how="left",
        # Now, since the left DataFrame already contains "_home" columns,
        # we use suffixes=("_home", "_away") to ensure this merge adds distinct "_away" columns
        suffixes=("_home", "_away")
    )
    
    # Now that the data has been merged, we can drop redundant team_home and team_away columns.
    df = df.drop(columns=["team_home", "team_away"])
    
    # Since we're using rolling averages, the first n_matches games will have NaN values, so drop them.
    # Only drop if both teams has missing data, since dropping rows hurts debugging.
    # df = df.dropna(subset=["form_goals_scored_home", "form_goals_scored_away"], how="all").reset_index(drop=True)
    
    rolling_cols = [c for c in df.columns if c.startswith("form_")]
    df = df.dropna(subset=rolling_cols).reset_index(drop=True)
        
    return df

# Compute win streak
def consecutive_win_streak_before(wins: pd.Series) -> pd.Series:
    """
    Returns the number of consecutive wins before each game.
    Example: wins = [1,1,0,1] -> streak = [0,1,0,0]
    """
    prev = wins.shift(1).fillna(0).astype(int)   # only look at prior results (leak-free)
    # Vectorized run-length cumsum over blocks separated by zeros
    groups = (prev == 0).cumsum()
    return prev.groupby(groups).cumsum()


def add_elo_features(
    df: pd.DataFrame,
    K: float = 24.0,
    base: float = 1500.0,
    home_adv: float = 60.0,
    season_regress: float = 0.25,
) -> pd.DataFrame:
    """
    Adds pre-match Elo features:
        - elo_home_pre, elo_away_pre, elo_diff_pre
    And also keeps post-match ratings for debugging (elo_home_post, elo_away_post).

    Parameters
    ----------
    K : float
        Elo K-factor (update size). Typical 16–32 for soccer.
    base : float
        Starting rating for all teams.
    home_adv : float
        Home-advantage rating bump (e.g., +60 Elo for the home team).
    season_regress : float in [0,1]
        At a new season, ratings := (1 - season_regress) * old + season_regress * base.
        Set to 0.0 to disable regression.

    Returns
    -------
    df : DataFrame with added Elo columns (pre-match features).
    """
    # Work on a copy, sorted chronologically per your pipeline
    df = df.sort_values(["season", "date"]).reset_index(drop=True).copy()

    # Storage for output columns
    elo_home_pre, elo_away_pre = [], []
    elo_home_post, elo_away_post = [], []

    # Ratings dict, reset / regressed each season
    current_season = None
    ratings = {}

    for idx, row in df.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]  # 'H', 'D', or 'A'

        # When season changes, optionally regress everyone toward base
        if season != current_season:
            if current_season is not None and season_regress > 0.0:
                for t in ratings.keys():
                    ratings[t] = (1 - season_regress) * ratings[t] + season_regress * base
            current_season = season

        # Ensure teams exist in ratings dict
        if home not in ratings:
            ratings[home] = base
        if away not in ratings:
            ratings[away] = base

        Rh, Ra = ratings[home], ratings[away]

        # Pre-match ratings (what you’ll actually use as features)
        elo_home_pre.append(Rh)
        elo_away_pre.append(Ra)

        # Expected score for home with home-adv bump
        # E_home = 1 / (1 + 10^((Ra - (Rh + home_adv))/400))
        Rh_adj = Rh + home_adv
        exp_home = 1.0 / (1.0 + 10.0 ** ((Ra - Rh_adj) / 400.0))
        exp_away = 1.0 - exp_home

        # Actual scores
        if result == "H":
            s_home, s_away = 1.0, 0.0
        elif result == "A":
            s_home, s_away = 0.0, 1.0
        else:  # 'D'
            s_home, s_away = 0.5, 0.5

        # Elo updates
        Rh_new = Rh + K * (s_home - exp_home)
        Ra_new = Ra + K * (s_away - exp_away)

        ratings[home] = Rh_new
        ratings[away] = Ra_new

        elo_home_post.append(Rh_new)
        elo_away_post.append(Ra_new)

    # Attach to df
    df["elo_home_pre"] = np.array(elo_home_pre, dtype=float)
    df["elo_away_pre"] = np.array(elo_away_pre, dtype=float)
    df["elo_diff_pre"] = df["elo_home_pre"] - df["elo_away_pre"]

    return df
