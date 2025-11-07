# Dev Note
@Max all pre-processing will be done in the src/data directory. First load_data loads the data and only considers the columns we want, and then it's sent to build_features.py where we deal with computing features based on the last 5 games. This can be seen in main.py as well.

# TODO
- Incorporate bookmaker odds (I chose in config to use Bet365)

# Areas to improve on
- Streak has not yet been implemented in build_features.py
- Perhaps # wins is not perfect since maybe the team has 0 wins but drew 5 times or something