# Bagged trees; very robust baseline. No scaling needed.
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_model(X_train,y_train):

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=600,       # more trees -> lower variance
            max_depth=None,         # let trees expand; we control with min_samples_leaf
            min_samples_leaf=2,     # regularization to avoid overfit on tiny leaves
            class_weight="balanced",# help minority class (often draws)
            random_state=0
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline