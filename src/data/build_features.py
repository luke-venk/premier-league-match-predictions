"""
Compute rolling stats to reflect a team's form.
"""

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
        for col in ["wins","points", "goals_scored", "goals_conceded", "shots_on_target", "fouls_committed"]:
            # shift() prevents data leakage by shifting the current row down and only including prior rows.
            # rolling(n_matches) creates rolling window of n_matches entries
            team_df[f"form_{col}"] = team_df[col].shift().rolling(n_matches).sum()
            
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