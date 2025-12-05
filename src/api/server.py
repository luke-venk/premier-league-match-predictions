"""
Flask API server that handles HTTP requests and routing.
"""
from flask import Flask

from src.api.inference import Predictor
from src.config import Models
from src.models.train_model import train

# Global configuration.
app = Flask(__name__)

# Predictor class handles heavyweight computations and predictions.
predictor = Predictor("models/voting_model.joblib")

@app.route("/")
def hello_world():
    return "Hello, world!\n"

# Start the web server.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")