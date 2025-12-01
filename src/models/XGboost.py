# Gradient-boosted trees (great for tabular). No scaling needed.
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.util.DFImputer import DFImputer

def train_model(X_train,y_train):

    pipeline =  Pipeline([
        ("imputer", DFImputer(strategy="median")),
        ("clf",  XGBClassifier(
            objective="multi:softprob",  # multiclass probabilities
            num_class=3,                 # 3 outcomes: H/A/D
            n_estimators=300,            # Number of boosted trees (tuned)
            max_depth=3,                 # Shallower trees generalize better (tuned)
            learning_rate=0.05,          # Shrinkage (tuned)
            subsample=0.9,               # Row subsampling (tuned)
            colsample_bytree=0.8,        # Column subsampling (tuned)
            reg_lambda=1.0,              # L2 regularization (tuned)
            tree_method="hist",          # Fast histogram-based
            eval_metric="mlogloss",
            random_state=0,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
