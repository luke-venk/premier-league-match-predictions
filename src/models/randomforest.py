# Bagged trees; very robust baseline. No scaling needed.
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.util.DFImputer import DFImputer

def train_model(X_train,y_train):

    pipeline = Pipeline([
        ("imputer", DFImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=600,        # More trees -> lower variance (tuned)
            max_depth=20,            # Let trees expand; we control with min_samples_leaf (tuned)
            min_samples_leaf=1,      # Regularization to avoid overfit on tiny leaves (tuned)
            class_weight="balanced", # Help minority class (often draws)
            random_state=0
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline