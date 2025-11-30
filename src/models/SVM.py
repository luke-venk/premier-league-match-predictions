from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from src.util.DFImputer import DFImputer
from sklearn.preprocessing import StandardScaler

def train_model(X_train, y_train):

    pipeline = Pipeline([
        ("imputer", DFImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=2.0,                # regularization strength
            gamma="scale",        # RBF width (auto)
            probability=True,     # enable predict_proba (needed for log loss/Brier)
            class_weight="balanced", # guard against class imbalance
            random_state=0
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline