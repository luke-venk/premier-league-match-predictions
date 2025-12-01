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
            n_estimators=600,            # number of boosted trees
            max_depth=4,                 # shallower trees generalize better
            learning_rate=0.05,          # shrinkage
            subsample=0.9,               # row subsampling
            colsample_bytree=0.9,        # column subsampling
            reg_lambda=1.0,              # L2 regularization
            tree_method="hist",          # fast histogram-based
            eval_metric="mlogloss",
            random_state=0,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
