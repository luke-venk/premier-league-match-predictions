# Developing a Model to Predict Premier League Match Outcomes
## Project Overview
The English Premier League (EPL) is by far the most watched sports league in the world, attracting a global fanbase in the hundreds of millions every year. Football (soccer) is the world’s game, and over the years data-driven analysis has permanently altered the sport’s landscape. The goal of this project is to build supervised machine learning systems that predict match outcomes using rolling team performance statistics, bookmaker odds, and supplemental variables such as squad valuations.  

While predicting football results is inherently uncertain, this project demonstrates how data-driven models can quantify competitive form using real-world sports data. Beyond its entertainment value, our framework illustrates practical machine learning workflows - from raw data ingestion and preprocessing, to feature engineering, to training and evaluation - in a reproducible, usable manner.

## Data
We are using [Football-Data.co.uk](https://football-data.co.uk/englandm.php) as our primary data source for Premier League match statistics. Each CSV summarizes several match statistics (match outcome, goals, shots, etc.) for all 380 matches in a Premier League season, for both the home and away teams. Datasets date as far back as the 1993/1994 season. We will default to using the previous 10 seasons from our most current, complete dataset. In this case, the beginning season will be the 2015/2016 season, and the end season will be the 2024/2025 season. We use a temporal 70/30 train-test split since a season naturally has a chronological ordering to it, using the first 7 seasons for training, and reserving the 3 most recent seasons for testing.  

We will also use other data sources to supplement our main data source. For example, we will use [TransferMarkt](https://www.transfermarkt.co.uk/premier-league/startseite/wettbewerb/GB1) to gather information about total squad value for a given season.  

## Set up virtual environment
To deal with dependencies, create a virtual environment (ignored from repository):  
`python3 -m venv venv`  

Activate the virtual environment:  
`source venv/bin/activate`  

Download the required packages into the virtual environment:  
`pip3 install -r requirements.txt`

Some Unix machines may not support OpenMP runtime libomp.dylib for XGboost and lightGBM easiest fix is to homebrew install and ensure loader can see it
in bash

python3 -m venv .venv && source .venv/bin/activate
brew install libomp
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
pip install --upgrade pip
pip3 install -r requirements.txt

alternatively in train_model.py comment out lines 7 and 12 and don't use XGboost or lightGBM

## Features
Given any match, we will engineer features over the last N games a team has played, which indicates a team's form, which is very relevant in soccer. We then consider the following features for both the home and away team in a given match, over the last N games:  
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
│   ├── raw/                       # Original CSVs (as downloaded)  
│   ├── processed/                 # Cleaned and feature-engineered datasets  
│   ├── test/                      # CSVs used for simple unit tests
│  
├── notebooks/                     # Exploratory work and visualization  
│  
├── src/                           # Library of modules  
│   ├── data/  
│   │   ├── load_data.py           # Load & clean raw CSVs into DataFrames  
│   │   ├── build_features.py      # Compute features related to form  
│   │   └── test_load_and_build.py # Unit tests for feature engineering
│   ├── models/  
│   │   ├── train_model.py         # Train classifier (LogReg/XGBoost/etc.)  
│   │   └── evaluate_model.py      # Metrics, confusion matrix, etc.  
│   └── config.py                  # Configs like number of matches for form, etc.  
│  
├── main.py                        # Entry point: runs the full pipeline  
├── requirements.txt  
├── README.md  
└── .gitignore  

## Testing
A series of unit tests has been written to confirm our data loading and feature building functions as intended. To run them, go to the project root directory, and run the following:  
`pytest`  

The CSV files used for unit tests are found in data/test/. The names of the files should be indicative of what case is being checked for. The substring `n#` indicates what the value of `n_matches` is for detecting form, which immediately impacts the calculations of form statistics.