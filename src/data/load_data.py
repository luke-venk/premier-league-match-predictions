"""
Read and clean the raw CSVs from our data source, and load into a Pandas DataFrame.
"""
import re
import pandas as pd

def load_matches(csv_path: str, sportsbook: str) -> pd.DataFrame:
    """
    Load our data from the CSVs to a Pandas DataFrame for our model to use. The point
    of this is to only load the columns from the dataset that we want:
        - Home Team
        - Away Team
        - Full Time Result (FTR)
        - Full Time Home Team Goals (FTHG)
        - Full Time Away Team Goals (FTAG)
        - Home Team Shots on Target (HST)
        - Away Team Shots on Target (AST)
        - Home Team Fouls Committed (HF)
        - Away Team Fouls Committed (AF)
    
    All the features that we want to engineer (see build_features.py) will be engineered
    in that module.
    
    Could possibly be extended to curl the dataset from online if the raw data
    for the configured season has not already been loaded:
        ex: curl -o data_24_25.csv https://football-data.co.uk/mmz4281/2425/E0.csv    
    
    Args:
        csv_path: The path to the CSV.
        sportsbook: The acronym of the betting company whose odds we want to use.
    
    Returns:
        A DataFrame with our data.
    """
    df = pd.read_csv(csv_path)
    
    # See list of abbreviations for the dataset at the following link:
    # https://football-data.co.uk/notes.txt

    
    if str(sportsbook).lower() == "aggregate":
        odds_home_cols = [c for c in df.columns if re.fullmatch(r"[A-Z0-9]{2,4}H", c)]
        odds_draw_cols = [c for c in df.columns if re.fullmatch(r"[A-Z0-9]{2,4}D", c)]
        odds_away_cols = [c for c in df.columns if re.fullmatch(r"[A-Z0-9]{2,4}A", c)]

        if not (odds_home_cols and odds_draw_cols and odds_away_cols):
            raise ValueError("Could not find sportsbook odds columns to aggregate (H/D/A).")

        # row-wise mean (ignore NaNs), coerce any stray strings to NaN
        df["odds_home_win"] = df[odds_home_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        df["odds_draw"]     = df[odds_draw_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        df["odds_away_win"] = df[odds_away_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

        # drop all original odds columns
        df.drop(columns=list(set(odds_home_cols + odds_draw_cols + odds_away_cols)), inplace=True)

        # Now rename the *non-odds* columns as usual, and keep the 3 aggregated odds
        rename_map_base = {
            # independent vars
            "Date":     "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG":     "home_goals",
            "FTAG":     "away_goals",
            "HST":      "home_shots_on_target",
            "AST":      "away_shots_on_target",
            "HF":       "home_fouls",
            "AF":       "away_fouls",
            # dependent var
            "FTR": "result",
        }

        df = df[list(rename_map_base.keys()) + ["odds_home_win", "odds_draw", "odds_away_win"]]
        df = df.rename(columns=rename_map_base)

    else:
        rename_map = {
            # Independent variables
            'Date':           'date',
            'HomeTeam':       'home_team',
            'AwayTeam':       'away_team',
            'FTHG':           'home_goals',
            'FTAG':           'away_goals',
            'HST':            'home_shots_on_target',
            'AST':            'away_shots_on_target',
            'HF':             'home_fouls',
            'AF':             'away_fouls',
            f'{sportsbook}H': 'odds_home_win',
            f'{sportsbook}D': 'odds_draw',
            f'{sportsbook}A': 'odds_away_win',
            
            # Dependent variable
            'FTR':            'result'
        }
        
        # Only keep the columns whose keys are in the rename map.
        df = df[list(rename_map.keys())]
        # Rename the columns from the keys to the values.
        df = df.rename(columns=rename_map)
    
    # Parse date column into datetime.
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    
    # Drop rows without a valid result (e.g., postponed, or not home, draw, or away).
    df = df.dropna(subset=['result'])
    df = df[df['result'].isin(['H', 'D', 'A'])]
    
    return df