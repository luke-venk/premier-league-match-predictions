# Leaf-wise gradient boosting—excellent on tabular data.
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from src.util.DFImputer import DFImputer

def train_model(X_train, y_train):

    pipeline =  Pipeline([
        ("imputer", DFImputer(strategy="median")),
        ("clf", lgb.LGBMClassifier(
            objective="multiclass", # multiclass probabilities
            num_class=3,        # 3 outcomes: H/A/D
            n_estimators=600,       # number of boosted trees
            learning_rate=0.05,     # shrinkage
            num_leaves=30,        # complexity control
            subsample=0.9,          # row subsampling
            colsample_bytree=0.9,    # column subsampling
            reg_lambda=1.0,          #L2 regularization
            random_state=0,  # Random seed
            verbose=-1  # No outputs
    ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

