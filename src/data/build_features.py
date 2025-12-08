"""
Compute rolling stats to reflect a team's form.
"""
import numpy as np
import pandas as pd
from src.config import N_MATCHES
from src.data.scrape_values import merge_valuations_into_dataframe

def build_rolling_features(
    df: pd.DataFrame, 
    n_matches: int = N_MATCHES, 
    elo: bool = None, 
    h2h: bool = None, 
    diff: bool = None,
    delete_original_diff: bool = None
) -> pd.DataFrame:
    """
    Compute the rolling stats of the following features both home and away teams:
        - Wins (last 5 games)
        - Points (last 5 games)
        - Goals scored (last 5 games)
        - Goals conceded (last 5 games)
        - Shots on target (last 5 games)
        - Fouls committed (last 5 games)
        - Possession percentage (last 5 games)
        - Head-to-head history (last 5 games)
        - Win streak length
        - Bookmaker odds (for this match)
        - Difference features (home - away)
    
    Args:
        df: Our raw DataFrame.
        n_matches: Number of matches for rolling window.
        elo: Enable Elo features (None = use config default).
        h2h: Enable H2H features (None = use config default).
        diff: Enable difference features (None = use config default).
        delete_original_diff: Delete original home/away columns when using diff (None = use config default).
    
    Returns:
        A DataFrame with our pre-processed features.
    """
    # Import config here to avoid circular imports
    from src.config import USE_ELO, USE_H2H, USE_DIFF, DELETE_ORIGINAL_DIFF
    
    # Use config defaults if not explicitly provided
    if elo is None:
        elo = USE_ELO
    if h2h is None:
        h2h = USE_H2H
    if diff is None:
        diff = USE_DIFF
    if delete_original_diff is None:
        delete_original_diff = DELETE_ORIGINAL_DIFF
    
    # Ensure data is chronologically sorted (although, it already should be).
    # Adding 'season' as a grouping key prevents rolling windows from crossing
    # season boundaries.
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    if elo:
        # Add ELo: Season regress starts off new seasons by returning elo closer to base.
        df = add_elo_features(df, K=24.0, base=1500.0, home_adv=60.0, season_regress=0.25)
    
    if h2h:
        # Add head-to-head history features
        df = add_h2h_features(df, n_h2h_matches=n_matches)
    
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
        
        # Possession percentage
        team_df["possession_pct"] = team_df["home_possession_pct"].where(mask_home, team_df["away_possession_pct"])

        # Win streak
        team_df["form_win_streak"] = consecutive_win_streak_before(team_df["wins"])
        
        # For each of the metrics we just computed, calculate the total metrics over the
        # previous n_matches games.
        for col in ["wins", "points", "goals_scored", "goals_conceded", "shots_on_target", "fouls_committed"]:
            # shift() prevents data leakage by shifting the current row down and only including prior rows.
            # rolling(n_matches) creates rolling window of n_matches entries
            team_df[f"form_{col}"] = team_df[col].shift().rolling(n_matches, min_periods=1).sum()
        
        # For possession_pct, use mean instead of sum (more meaningful for percentages)
        team_df["form_possession_pct"] = team_df["possession_pct"].shift().rolling(n_matches, min_periods=1).mean()
            
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
                "form_win_streak",
                "form_possession_pct"
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
    
    # Add difference features (home - away) for improved PCA and interpretability
    if diff:
        df = add_diff_features(df, delete_original=delete_original_diff)
        
    return df


# Curated feature list for FEWER_FEATURES option (only available when USE_DIFF=True)
FEWER_FEATURES_LIST = [
    'form_possession_pct_diff',
    'elo_diff_pre',
    'form_goals_conceded_diff',
    'form_win_streak_diff',
    'form_shots_on_target_diff',
    'odds_home_win',
    'odds_away_win',
    'odds_draw',
    'form_goals_scored_diff',
    'form_wins_diff',
    'h2h_draws'
]


def get_feature_columns(df_columns):
    """
    Dynamically construct the list of feature columns based on configuration.
    
    Args:
        df_columns: List of all columns in the dataframe
        
    Returns:
        List of feature column names to use in the model
    """
    from src.config import USE_ELO, USE_H2H, USE_DIFF, DELETE_ORIGINAL_DIFF, FEWER_FEATURES
    
    # If FEWER_FEATURES is enabled, use the curated subset (only works with USE_DIFF=True)
    if FEWER_FEATURES:
        if not USE_DIFF:
            raise ValueError("FEWER_FEATURES requires USE_DIFF=True")
        feature_cols = [c for c in FEWER_FEATURES_LIST if c in df_columns]
        return feature_cols
    
    feature_cols = []
    
    # Always include form features (unless deleted by diff)
    if USE_DIFF and DELETE_ORIGINAL_DIFF:
        # Only include diff features, not individual home/away
        feature_cols += [c for c in df_columns if c.startswith('form_') and c.endswith('_diff')]
    else:
        # Include all form features
        feature_cols += [c for c in df_columns if c.startswith('form_')]
    
    # Add Elo features if enabled
    if USE_ELO:
        if USE_DIFF and DELETE_ORIGINAL_DIFF:
            # Only include diff
            if 'elo_diff_pre' in df_columns:
                feature_cols.append('elo_diff_pre')
        else:
            # Include all Elo features
            if 'elo_home_pre' in df_columns:
                feature_cols.append('elo_home_pre')
            if 'elo_away_pre' in df_columns:
                feature_cols.append('elo_away_pre')
            if 'elo_diff_pre' in df_columns:
                feature_cols.append('elo_diff_pre')
    
    # Add H2H features if enabled
    if USE_H2H:
        if USE_DIFF and DELETE_ORIGINAL_DIFF:
            # Only include h2h diff features and aggregate features
            feature_cols += [c for c in df_columns if c.startswith('h2h_') and 
                           (c.endswith('_diff') or c in ['h2h_matches', 'h2h_draws'])]
        else:
            # Include all H2H features
            feature_cols += [c for c in df_columns if c.startswith('h2h_')]
    
    # Always include odds (they're already relative to outcomes)
    feature_cols += ['odds_home_win', 'odds_draw', 'odds_away_win']
    
    # Always include possession features (these are pre-match aggregates)
    if 'home_possession_pct' in df_columns:
        feature_cols.append('home_possession_pct')
    if 'away_possession_pct' in df_columns:
        feature_cols.append('away_possession_pct')
    if 'possession_diff' in df_columns:
        feature_cols.append('possession_diff')
    
    # Filter to only columns that actually exist
    feature_cols = [c for c in feature_cols if c in df_columns]
    
    return feature_cols


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


def add_h2h_features(df: pd.DataFrame, n_h2h_matches: int) -> pd.DataFrame:
    """
    Add head-to-head history features for each match based on previous encounters
    between the same two teams.
    
    For each match, this computes statistics from the last N matches between
    the home team and away team, including:
        - Total H2H matches played
        - Home team wins in H2H
        - Away team wins in H2H
        - Draws in H2H
        - Home team goals scored in H2H
        - Home team goals conceded in H2H
        - Away team goals scored in H2H
        - Away team goals conceded in H2H
        
    This is done in a leak-free manner by only considering matches that occurred
    before the current match date.
    
    Parameters
    ----------
    df : pd.DataFrame
        The match dataframe sorted by season and date.
    n_h2h_matches : int
        Number of previous H2H matches to consider (default 5).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added H2H features.
    """
    # Ensure sorted
    df = df.sort_values(["season", "date"]).reset_index(drop=True).copy()
    
    # Initialize columns for H2H features
    h2h_features = {
        'h2h_matches': [],
        'h2h_home_wins': [],
        'h2h_away_wins': [],
        'h2h_draws': [],
        'h2h_home_goals_scored': [],
        'h2h_home_goals_conceded': [],
        'h2h_away_goals_scored': [],
        'h2h_away_goals_conceded': [],
        'h2h_home_win_pct': [],
        'h2h_away_win_pct': []
    }
    
    # For each match, compute H2H statistics
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        current_date = row['date']
        
        # Find all previous matches between these two teams (in either order)
        # Only look at matches before the current date to prevent data leakage
        h2h_matches = df[
            (
                ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                ((df['home_team'] == away_team) & (df['away_team'] == home_team))
            ) &
            (df['date'] < current_date)
        ].tail(n_h2h_matches).copy()
        
        # If no previous H2H matches, fill with zeros
        if len(h2h_matches) == 0:
            h2h_features['h2h_matches'].append(0)
            h2h_features['h2h_home_wins'].append(0)
            h2h_features['h2h_away_wins'].append(0)
            h2h_features['h2h_draws'].append(0)
            h2h_features['h2h_home_goals_scored'].append(0)
            h2h_features['h2h_home_goals_conceded'].append(0)
            h2h_features['h2h_away_goals_scored'].append(0)
            h2h_features['h2h_away_goals_conceded'].append(0)
            h2h_features['h2h_home_win_pct'].append(0.0)
            h2h_features['h2h_away_win_pct'].append(0.0)
            continue
        
        # Count total H2H matches
        n_h2h = len(h2h_matches)
        
        # For each H2H match, determine which team was home and which was away
        # and compute outcomes from the perspective of the CURRENT match's home/away teams
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals_scored = 0
        home_goals_conceded = 0
        away_goals_scored = 0
        away_goals_conceded = 0
        
        for _, h2h_row in h2h_matches.iterrows():
            h2h_home = h2h_row['home_team']
            h2h_away = h2h_row['away_team']
            h2h_result = h2h_row['result']
            h2h_home_goals = h2h_row['home_goals']
            h2h_away_goals = h2h_row['away_goals']
            
            # Check if current home team was home or away in this H2H match
            if h2h_home == home_team:
                # Current home team was playing at home in H2H match
                home_goals_scored += h2h_home_goals
                home_goals_conceded += h2h_away_goals
                away_goals_scored += h2h_away_goals
                away_goals_conceded += h2h_home_goals
                
                if h2h_result == 'H':
                    home_wins += 1
                elif h2h_result == 'A':
                    away_wins += 1
                else:
                    draws += 1
            else:
                # Current home team was playing away in H2H match
                home_goals_scored += h2h_away_goals
                home_goals_conceded += h2h_home_goals
                away_goals_scored += h2h_home_goals
                away_goals_conceded += h2h_away_goals
                
                if h2h_result == 'A':
                    home_wins += 1
                elif h2h_result == 'H':
                    away_wins += 1
                else:
                    draws += 1
        
        # Calculate win percentages
        home_win_pct = home_wins / n_h2h if n_h2h > 0 else 0.0
        away_win_pct = away_wins / n_h2h if n_h2h > 0 else 0.0
        
        # Store features
        h2h_features['h2h_matches'].append(n_h2h)
        h2h_features['h2h_home_wins'].append(home_wins)
        h2h_features['h2h_away_wins'].append(away_wins)
        h2h_features['h2h_draws'].append(draws)
        h2h_features['h2h_home_goals_scored'].append(home_goals_scored)
        h2h_features['h2h_home_goals_conceded'].append(home_goals_conceded)
        h2h_features['h2h_away_goals_scored'].append(away_goals_scored)
        h2h_features['h2h_away_goals_conceded'].append(away_goals_conceded)
        h2h_features['h2h_home_win_pct'].append(home_win_pct)
        h2h_features['h2h_away_win_pct'].append(away_win_pct)
    
    # Add all H2H features to the dataframe
    for feature_name, feature_values in h2h_features.items():
        df[feature_name] = feature_values
    
    return df


def add_diff_features(df: pd.DataFrame, delete_original: bool = False) -> pd.DataFrame:
    """
    Add difference (home - away) features to the dataframe.
    Only creates differences for features that exist in the dataframe.
    """
    df = df.copy()
    
    # Track columns to potentially delete later
    cols_to_delete = []
    
    # Elo diff (may already exist)
    if 'elo_home_pre' in df.columns and 'elo_away_pre' in df.columns:
        if 'elo_diff_pre' not in df.columns:
            df['elo_diff_pre'] = df['elo_home_pre'] - df['elo_away_pre']
        if delete_original:
            cols_to_delete.extend(['elo_home_pre', 'elo_away_pre'])
    
    # Form diffs (always present)
    df['form_wins_diff'] = df['form_wins_home'] - df['form_wins_away']
    df['form_points_diff'] = df['form_points_home'] - df['form_points_away']
    df['form_goals_scored_diff'] = df['form_goals_scored_home'] - df['form_goals_scored_away']
    df['form_goals_conceded_diff'] = df['form_goals_conceded_home'] - df['form_goals_conceded_away']
    df['form_shots_on_target_diff'] = df['form_shots_on_target_home'] - df['form_shots_on_target_away']
    df['form_fouls_committed_diff'] = df['form_fouls_committed_home'] - df['form_fouls_committed_away']
    df['form_win_streak_diff'] = df['form_win_streak_home'] - df['form_win_streak_away']
    df['form_possession_pct_diff'] = df['form_possession_pct_home'] - df['form_possession_pct_away']
    
    if delete_original:
        cols_to_delete.extend([
            'form_wins_home', 'form_wins_away',
            'form_points_home', 'form_points_away',
            'form_goals_scored_home', 'form_goals_scored_away',
            'form_goals_conceded_home', 'form_goals_conceded_away',
            'form_shots_on_target_home', 'form_shots_on_target_away',
            'form_fouls_committed_home', 'form_fouls_committed_away',
            'form_win_streak_home', 'form_win_streak_away',
            'form_possession_pct_home', 'form_possession_pct_away'
        ])
    
    # H2H diffs (only if H2H features exist)
    if 'h2h_home_wins' in df.columns and 'h2h_away_wins' in df.columns:
        df['h2h_wins_diff'] = df['h2h_home_wins'] - df['h2h_away_wins']
        df['h2h_goals_scored_diff'] = df['h2h_home_goals_scored'] - df['h2h_away_goals_scored']
        df['h2h_goals_conceded_diff'] = df['h2h_home_goals_conceded'] - df['h2h_away_goals_conceded']
        df['h2h_win_pct_diff'] = df['h2h_home_win_pct'] - df['h2h_away_win_pct']
        
        if delete_original:
            cols_to_delete.extend([
                'h2h_home_wins', 'h2h_away_wins',
                'h2h_home_goals_scored', 'h2h_away_goals_scored',
                'h2h_home_goals_conceded', 'h2h_away_goals_conceded',
                'h2h_home_win_pct', 'h2h_away_win_pct'
            ])
    
    # Delete original columns if requested (only ones that exist)
    if delete_original and cols_to_delete:
        existing_cols_to_delete = [c for c in cols_to_delete if c in df.columns]
        if existing_cols_to_delete:
            df = df.drop(columns=existing_cols_to_delete)
    
    return df


def get_diff_feature_columns() -> list:
    """
    Get the list of diff feature column names for modeling.
    Note: This is deprecated. Use get_feature_columns() instead for dynamic selection.
    """
    from src.config import USE_ELO, USE_H2H
    
    features = [
        # Odds (keep as is - they're already relative to outcomes)
        "odds_home_win", "odds_draw", "odds_away_win",
        
        # Form diffs (always present)
        "form_wins_diff", "form_points_diff", "form_goals_scored_diff",
        "form_goals_conceded_diff", "form_shots_on_target_diff",
        "form_fouls_committed_diff", "form_win_streak_diff",
        "form_possession_pct_diff",
    ]
    
    # Elo diff (only if enabled)
    if USE_ELO:
        features.append("elo_diff_pre")
    
    # H2H diffs (only if enabled - keep totals: matches, draws)
    if USE_H2H:
        features.extend([
            "h2h_matches", "h2h_draws",
            "h2h_wins_diff", "h2h_goals_scored_diff", 
            "h2h_goals_conceded_diff", "h2h_win_pct_diff",
        ])
    
    return features
