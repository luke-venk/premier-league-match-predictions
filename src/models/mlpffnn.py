# Simple feed-forward network for tabular data.
# Needs scaling; Adam optimizer built-in. MLPs train better with standard features
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from src.util.DFImputer import DFImputer
from sklearn.preprocessing import StandardScaler

def train_model(X_train, y_train):
    
    pipeline = Pipeline([
        ("imputer", DFImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # Two-layer FFN (tuned)
            activation="tanh",             # Activation function (tuned)
            solver="adam",                 # Adam optimizer
            alpha=0.01,                  # L2 weight decay (tuned)
            learning_rate_init=0.01,       # Learning rate (tuned)
            max_iter=5000,
            random_state=0
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline