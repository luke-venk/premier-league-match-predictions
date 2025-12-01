from sklearn.pipeline import Pipeline
from src.util.DFImputer import DFImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_model(X_train,y_train):
    # Define list of estimators to include in voting ensemble.
    estimators = []

    # 1) Logistic Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()), # Must be scaled
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            C=1.0,
            random_state=0
        ))
    ])
    estimators.append(("lr", lr_pipe))

    # 2) Random Forest
    rf_pipe = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0
        ))
    ])
    estimators.append(("rf", rf_pipe))

    # 3) HistGradientBoosting
    hgb_pipe = Pipeline([
        ("clf", HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_depth=6,
            max_iter=600,
            random_state=0
        ))
    ])
    estimators.append(("hgb", hgb_pipe))

    # 4) XGBoost
    xgb_pipe = Pipeline([
        ("clf", xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=500,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=0,
            eval_metric="mlogloss",
            verbosity=0
        ))
    ])
    estimators.append(("xgb", xgb_pipe))

    # We can bias the vote by passing weights=[...]; else equal weights
    vc = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("imputer", DFImputer(strategy="median")),
        ("clf", vc)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
