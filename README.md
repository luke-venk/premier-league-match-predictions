# Premier League Match Predictions
## Overview
Soccer, or football as it will be referred to in this paper, is the world's sport. The English Premier League is the most popular sports league in the world, garnering billions of views every year. In recent years, the abundance of football data and advancements in machine learning have revolutionized the use of predictive models for football outcomes. In this paper, we seek to compare the performance of various machine learning models at predicting Premier League match outcomes. This is a classification problem with 3 possible outcomes: a home team win, an away team win, or a draw. 

Before selecting and training our models, it is important to consider the inherent difficulty of this prediction task. Due to the nature of the low-scoring sport luck and fine margins significantly impact results, causing match outcomes to be notoriously noisy. Furthermore, a team's recent performance, or "form", strongly influences match outcomes; a big challenge for us was figuring out how to quantify form. Finally, data sources encapsulating all the features we might wish to use in our feature matrix may be difficult to access, either due to paywall or due to difficulty merging datasets.

This study aims to improve upon prior approaches of using machine learning for predicting football results by yielding a high accuracy while only using data from previous matches. Many other studies have used statistics from the game they are currently predicting (i.e. score at half-time, game-possession, etc.). We want to avoid data leakage, so we aimed to develop a model that could make predictions based on data before the game.

## External Resources
This README is designed to be lightweight, providing the user with just the information the user needs to use the application. For a more detailed explanation of this project, please refer to one of the following:
- [Report](TODO)
- [Presentation Slides](https://docs.google.com/presentation/d/10yDmUe-KwvAvrPvzXGmXdAeF3LhDp9UKxyo9cHAXkhM/edit?slide=id.p#slide=id.p)
- [Video Demonstration](TODO)

## Data
We use a few data sources for this project. Firstly, we used [Football-Data.co.uk](https://football-data.co.uk/englandm.php) as our primary data source for match statistics (goals, shots, fouls, etc.). In addition, we used data from [TransferMarkt](https://www.transfermarkt.co.uk/premier-league/startseite/wettbewerb/GB1) to provide the estimated market valuation of the teams in a given match. Finally, we used [FootballCritic](https://www.footballcritic.com/premier-league/season-2025-2026/2/76035) to scrape possession data.

## Model
When creating our models, we included 10 seasons of data, from the 2015/2016 season to the most recently complete 2024/2025 season. We used a 70-30 chronological split, training our models on the seasons from 2015/2016 to 2021/2022, and evaluating performance using the seasons 2022/2023 to 2024/2025. After extensive analysis (can be found in our notebooks/ directory), we deployed our best performing model, which was **Logistic Regression** with N_MATCHES set to 5. The user will be able to interact with the model by using the inference server, so they can predict match outcomes for the latest season, the 2024/2025 Premier League season.

## Deploying the Inference Server
There are 2 ways the user can deploy the inference server:

### Pull Image from Docker Hub (TODO: update)
To quickly run the container using a prebuilt image pushed to Docker Hub, the user can run the following command:  
```docker pull lukevenk1/hurricane-inference:1.0```   

After the image has been pulled, to run the container, the user can run the following command:  
```docker run -p 5000:5000 lukevenk1/hurricane-inference:1.0```

### Build and Run Using the Source Code (TODO: update)
If the user would rather clone all the source code and rebuild the image locally, they can do so by cloning our repository and using Docker Compose.

For simplification, we use a Makefile to automate the building and deployment of the Docker container. If the user uses the Makefile, by default it will bring any existing container down and then restart it:  
```make```  

If the user wishes to rebuild the image (e.g., if they modified Dockerfile or requirements.txt which directly impacts the build), they can run:  
```make build```  

Finally, if the user wants to guarantee a completely fresh build, they can run the following command, although this will take significantly more time than the previous approaches:  
```make clean```

Doing any of the above 3 commands will start the container on the local machine and expose port 5000. This allows the user to interact with the inference server over the network using a REST API.

## Making Requests
Once the inference server is running, the user can make 3 types of requests to interact with our inference server.  

### 1. `/summary` (GET)
If the user would like a summary of the highest accuracies each model achieved with our data, they can send a GET request to /summary:  
```curl localhost:5000/summary```  
TODO: example

### 2. `/teams` (GET)
If the user would like a summary of all the current teams in the Premier League for the most current season (2024/2025), they can send a GET request to /teams:  
```curl localhost:5000/teams```  
TODO: example

### 3. `/inference` (POST)
```curl -X POST localhost:5000/inference -H "Content-Type: application/json" -d '{"home_team": <home_team>, "away_team": <away_team>}'```  
TODO: example


## Set up virtual environment (TODO: remove)
To deal with dependencies, create a virtual environment (ignored from repository):  
`python3 -m venv venv`  

Activate the virtual environment:  
`source venv/bin/activate`  

Download the required packages into the virtual environment:  
`pip3 install -r requirements.txt`

Some Unix machines may not support OpenMP runtime libomp.dylib for XGBoost. The easiest fix is to homebrew install and ensure loader can see it in bash:
```
python3 -m venv .venv && source .venv/bin/activate
brew install libomp
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
pip install --upgrade pip
pip3 install -r requirements.txt
```

Alternatively, don't use XGBoost train_model.py.

## Testing
A series of unit tests has been written to confirm our data loading and feature building functions as intended. To run them, go to the project root directory, and run the following:  
`pytest`  

The CSV files used for unit tests are found in data/test/. The names of the files should be indicative of what case is being checked for. The substring `n#` indicates what the value of `n_matches` is for detecting form, which immediately impacts the calculations of form statistics.