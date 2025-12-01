# Very fast probabilistic baseline. Works best when features are roughly Gaussian and independent.
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from src.util.DFImputer import DFImputer

def train_model(X_train, y_train):

    pipeline = Pipeline([
        ("imputer", DFImputer(strategy="median")),
        ("clf", GaussianNB(var_smoothing=1e-9))     # Tiny variance floor for stability (tuned)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline