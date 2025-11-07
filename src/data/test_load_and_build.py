"""Unit tests to test load_data.py and build_features.py."""
import pandas as pd
from src.data.load_data import load_matches
from src.data.build_features import build_rolling_features

TEST_DATA_DIR = 'data/test'

def test_home_only_goals_scored():
    """
    Verify basic rolling sum for goals scored. Liverpool will play 4 home games against
    the same team. If n_matches = 3, the 4th row should be a sum of 3 goals.
    """
    df = load_matches(csv_path=f'{TEST_DATA_DIR}/home_only_n3_goals.csv', sportsbook='B365')
    df = build_rolling_features(df=df, n_matches=3)
    
    # Get the rolling number of goals for each match, from both home and away games.
    mask_home = df["home_team"] == "Liverpool"
    df["liverpool_form_goals"] = df["form_goals_scored_home"].where(mask_home, df["form_goals_scored_away"])
    
    # The last row should have accummulated 3 goals.
    last_row = df.iloc[-1]
    assert last_row["liverpool_form_goals"] == 3, f"Expected 3 goals, but got {last_row['liverpool_form_goals']}"
    
def test_home_away_stats():
    """
    Verify basic rolling sum for goals scored. Manchester United will play 5 games against
    various teams, both home and away. We'll set n_matches = 4 and expect various statistics.
    """
    df = load_matches(csv_path=f'{TEST_DATA_DIR}/home_away_n4.csv', sportsbook='B365')
    df = build_rolling_features(df=df, n_matches=4)
    
    mask_home = df["home_team"] == "Manchester United"
    df["manu_form_wins"] = df["form_wins_home"].where(mask_home, df["form_wins_away"])
    df["manu_form_goals_scored"] = df["form_goals_scored_home"].where(mask_home, df["form_goals_scored_away"])
    df["manu_form_goals_conceded"] = df["form_goals_conceded_home"].where(mask_home, df["form_goals_conceded_away"])
    df["manu_form_shots_on_target"] = df["form_shots_on_target_home"].where(mask_home, df["form_shots_on_target_away"])
    df["manu_form_fouls_committed"] = df["form_fouls_committed_home"].where(mask_home, df["form_fouls_committed_away"])
    
    last_row = df.iloc[-1]
    assert last_row["manu_form_wins"] == 3, f"Expected 3 wins, but got {last_row['manu_form_wins']}"
    assert last_row["manu_form_goals_scored"] == 9, f"Expected 9 goals, but got {last_row['manu_form_goals_scored']}"
    assert last_row["manu_form_goals_conceded"] == 11, f"Expected 3 wins, but got {last_row['manu_form_goals_conceded']}"
    assert last_row["manu_form_shots_on_target"] == 18, f"Expected 3 wins, but got {last_row['manu_form_shots_on_target']}"
    assert last_row["manu_form_fouls_committed"] == 19, f"Expected 3 wins, but got {last_row['manu_form_fouls_committed']}"