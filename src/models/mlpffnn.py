# Simple feed-forward network for tabular data.
# Needs scaling; Adam optimizer built-in. MLPs train better with standard features
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def train_model(X_train, y_train):
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),  # two-layer FFN
            activation="relu",
            solver="adam",                # Adam optimizer
            alpha=1e-3,                   # L2 weight decay
            learning_rate_init=1e-2,
            max_iter=5000,
            random_state=0
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline