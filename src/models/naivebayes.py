# Very fast probabilistic baseline. Works best when features are roughly Gaussian and independent.
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_model(X_train, y_train):

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GaussianNB(var_smoothing=1e-9))     # tiny variance floor for stability
    ])

    pipeline.fit(X_train, y_train)
    return pipeline