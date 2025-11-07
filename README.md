# Developing a Model to Predict Premier League Match Outcomes
## Set up virtual environment
To deal with dependencies, create a virtual environment (ignored from repository):  
`python3 -m venv venv`  

Activate the virtual environment:  
`source venv/bin/activate`  

Download the required packages into the virtual environment:  
`pip3 install -r requirements.txt`

## Data
We are using [Football-Data.co.uk][https://football-data.co.uk/englandm.php] as our primary data source for Premier League match statistics. Each CSV summarizes several match statistics for all 380 matches in a Premier League season, for both the home and away teams.  

For a given season, we exclude the first 5 games of each team to reserve for form. We then use the first 70% of the dataset for training, and the remaining 30% for testing. We use a temporal train-test split since a season naturally has a chronological ordering to it.  

If the data for the season configured in config.py has already been loaded and processed, we will just use the processed data. Otherwise, we will have to download it and pre-process it before training our model with it.

## Features
Given any match, we will engineer features over the last 5 games a team has played, which indicates a team's form, which is very relevant in soccer. We then consider the following features for both the home and away team in a given match, over the last 5 games:  
- Wins
- Goals scored
- Goals conceded
- Shots on target
- Streak length
- Bookmaker odds
- Fouls committed

## Project Structure
premier-league-predictor/  
├── data/  
│   ├── raw/                     # Original CSVs (as downloaded)  
│   ├── processed/               # Cleaned and feature-engineered datasets  
│  
├── notebooks/                   # Exploratory work and visualization  
│  
├── src/                         # Library of modules  
│   ├── data/  
│   │   ├── load_data.py         # Load & clean raw CSVs into DataFrames  
│   │   ├── build_features.py    # Compute features related to form  
│   │   └── utils.py             # Shared helper functions  
│   ├── models/  
│   │   ├── train_model.py       # Train classifier (LogReg/XGBoost/etc.)  
│   │   └── evaluate_model.py    # Metrics, confusion matrix, etc.  
│   └── config.py                # Configs like number of matches for form, etc.  
│  
├── main.py                      # Entry point: runs the full pipeline  
├── requirements.txt  
├── README.md  
└── .gitignore  

## Testing
A series of unit tests has been written to confirm our data loading and feature building functions as intended. To run them, go to the project root directory, and run the following:  
`pytest`  

The CSV files used for unit tests are found in data/test/. The names of the files should be indicative of what case is being checked for. The substring `n#` indicates what the value of `n_matches` is for detecting form, which immediately impacts the calculations of form statistics.