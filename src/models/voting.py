"""Voting ensemble method using our 3 highest performing individual models."""

from sklearn.pipeline import Pipeline
from src.util.DFImputer import DFImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_model(X_train,y_train):
    # Define list of estimators to include in voting ensemble.
    estimators = []

    # 1) Logistic Regression
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='lbfgs',
            penalty='l2',
            C=0.1,
            max_iter=5000,
            random_state=0
        ))
    ])
    estimators.append(("lr", lr_pipe))

    # 2) Random Forest
    rf_pipe = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=600,
            max_depth=20,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=0
        ))
    ])
    estimators.append(("rf", rf_pipe))

    # 3) XGBoost
    xgb_pipe = Pipeline([
        ("clf", xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=0,
            n_jobs=-1
        ))
    ])
    estimators.append(("xgb", xgb_pipe))

    vc = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[3, 1, 1],   # Voting weights (tuned)
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("imputer", DFImputer(strategy="median")),
        ("clf", vc)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
