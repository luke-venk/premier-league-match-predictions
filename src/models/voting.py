from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb

def train_model(X_train,y_train):

    estimators = []

    # 1) Logistic Regression (good calibration baseline)
    lr = LogisticRegression(
        # multi_class="multinomial",
        solver="lbfgs",
        max_iter=5000,
        C=1.0,
        random_state=0,
    )
    estimators.append(("lr", lr))

    # 2) Random Forest (strong tabular baseline)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    )
    estimators.append(("rf", rf))

    # 3) HistGradientBoosting (fast, no OpenMP headaches)
    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_depth=6,
        max_iter=600,
        random_state=0,
    )
    estimators.append(("hgb", hgb))

    # 4) XGBoost
    xgbc = xgb.XGBClassifier(
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
    )
    estimators.append(("xgb", xgbc))

    # 5) LightGBM
    lgbm = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=600,
        learning_rate=0.06,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.0,
        random_state=0,
        verbosity=-1,
    )
    estimators.append(("lgbm", lgbm))

    # You can bias the vote by passing weights=[...]; else equal weights
    vc = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=None, #you can add weights to see which are the best
        n_jobs=-1,
        flatten_transform=False,
    )

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", vc),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
