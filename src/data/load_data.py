"""
Read and clean the raw CSVs from our data source, and load into a Pandas DataFrame.
"""
import re
import pandas as pd

def get_processed_path(end_year: int, num_seasons: int) -> str:
    """
    Given the relevant configuration parameters, determine what the name of the
    output CSV in data/processed would be. Used for determining whether or not
    the data has already been computed, in order to prevent repeating unnecessary
    data engineering.
    
    Args:
        end_year: The second year of the most recent season we want to include
        in our dataset.
        num_seasons: The total number of seasons we would like to aggregate.
        
    Returns:
        The path to the processed data.
    """
    # Example: if the ending year was 25, indicating the 24/25 season, if we used 10
    # seasons, the beginning season would be the 15/16 season. Thus, the processed
    # name would be `processed_15_25_n10` to indicate the starting and ending seasons,
    # as well as the number of years of data used.
    start_year = end_year - num_seasons
    return f'data/processed/processed_{start_year}_{end_year}_n{num_seasons}.csv'

def load_season(csv_path: str, sportsbook: str) -> pd.DataFrame:
    """
    For a single season, load the relevant features into a Pandas DataFrame,
    which will be aggregated with DataFrames from all relevant seasons, and
    then fed into our model.
    
    The point of this is to only load the columns from the dataset that we want:
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
            # Independent variables
            "Date":     "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG":     "home_goals",
            "FTAG":     "away_goals",
            "HST":      "home_shots_on_target",
            "AST":      "away_shots_on_target",
            "HF":       "home_fouls",
            "AF":       "away_fouls",
            
            # Dependent variable
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
    
    # Parse date column into datetime. Can't specify DD/MM/YYYY because for some reason
    # the 16/17 season uses DD/MM/YY.
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    
    # Drop rows without a valid result (e.g., postponed, or not home, draw, or away).
    df = df.dropna(subset=['result'])
    df = df[df['result'].isin(['H', 'D', 'A'])]
    
    return df


def load_all_seasons(end_year: int, num_seasons: int, sportsbook: str) -> pd.DataFrame:
    """
    Load the data from the proper number of raw CSVs into a processed CSV
    that stores the data aggregated from all the relevant seasons.
    
    Args:
        end_year: The second year of the most recent season we want to include
        in our dataset.
        num_seasons: The total number of seasons we would like to aggregate.
        
    Returns:
        A DataFrame with our aggregated data (also saved to data/processed).
    """
    # Store list of all DataFrames corresponding to each relevant season.
    all_dfs = []
    
    # Load all CSVs corresponding to relevant seasons. This will end at `end_year`
    # and begin at the year starting the season `num_seasons` seasons from `end_year`.
    
    # Example: if the ending year was 25, indicating the 24/25 season, if we used 10
    # seasons, the beginning season would be the 15/16 season.
    
    # e.g., from 15 to 24
    for y1 in range(end_year - num_seasons, end_year):
        y2 = y1 + 1
        raw_csv_path = f'data/raw/raw_{y1}_{y2}.csv'
        df = load_season(raw_csv_path, sportsbook)
        df['season'] = y1
        
        all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError('Failed to load the raw data.')
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by date across all seasons.
    df_all = df_all.sort_values(["season", "date"]).reset_index(drop=True)
    
    return df_all
        
        