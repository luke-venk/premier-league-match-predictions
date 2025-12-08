"""
Flask API server that handles HTTP requests and routing.
"""
from flask import Flask, jsonify, request

from src.api.inference import Predictor
from src.config import Models
from src.models.train_model import train

# Global configuration.
app = Flask(__name__)

# Predictor class handles heavyweight computations and predictions.
predictor = Predictor("models/voting_model.joblib")

@app.route("/summary", methods=["GET"])
def get_summary():
    """
    Outputs a summary of the highest training and testing accuracies
    each model achieved, from best performance to worst performance.
    """
    summary = [
        {
            "model": "voting_ensemble",
            "training_accuracy": 0.58605,
            "testing_accuracy": 0.59249,
        },
        {
            "model": "logistic_regression",
            "training_accuracy": 0.58681,
            "testing_accuracy": 0.59071,
        },
        {
            "model": "random_forest",
            "training_accuracy": 0.57072,
            "testing_accuracy": 0.58624,
        },
        {
            "model": "xgboost",
            "training_accuracy": 0.57225,
            "testing_accuracy": 0.57909,
        },
        {
            "model": "support_vector_machine",
            "training_accuracy": 0.52587,
            "testing_accuracy": 0.54424,
        },
        {
            "model": "multilayer_perceptron_ffn",
            "training_accuracy": 0.50939,
            "testing_accuracy": 0.52011,
        },
        {
            "model": "naive_bayes",
            "training_accuracy": 0.47145,
            "testing_accuracy": 0.47364,
        }
    ]
    return jsonify(summary), 200

@app.route("/teams", methods=["GET"])
def get_teams():
    """
    Endpoint that allows the users to query what teams were in the most
    recent completed Premier League season (2024/2025). This is useful
    so they know what teams they can make predictions with.
    """
    return jsonify(predictor.get_teams()), 200

@app.route("/predict", methods=["POST"])
def inference():
    """
    Given the home and away teams, build the feature matrix based on
    the available data, and use the best model to predict who would
    win the Premier League match.
    """
    # Ensure request is valid JSON.
    if not request.is_json:
        return jsonify({"error": "Request must be valid JSON"}), 400
    
    # Get home and away teams.
    data = request.get_json()
    home_team = data.get("home_team")
    away_team = data.get("away_team")
    
    # Ensure these fields were provided.
    if home_team is None or away_team is None:
        return jsonify({"error": 
            "The input you provided is not valid. "
            "Please provide both the 'home_team' and 'away_team' fields."
            }), 400
    
    # Ensure both teams are actually in the Premier League for the latest
    # season.
    if (home_team not in predictor.get_teams() or
        away_team not in predictor.get_teams()):
        return jsonify({"error": 
            "The teams you input are not valid. "
            "Please use /teams to determine what teams are "
            "in the Premier League for the latest season."
            }), 400
    
    # Use predictor logic.
    prediction, home_win_probability, draw_probability, away_win_probability = predictor.predict(home_team, away_team)
    
    return jsonify({
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "probabilities": {
            "home_win": home_win_probability,
            "draw": draw_probability,
            "away_win": away_win_probability
        }
    }), 200

# Start the web server.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")