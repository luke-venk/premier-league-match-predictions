import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from src.config import USE_ELO, USE_H2H, USE_DIFF, DELETE_ORIGINAL_DIFF

plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["axes.grid"] = True


# Cell 2: load data and define target / feature matrix

DATA_PATH = "data/processed/processed_15_25_n10.csv"  # change this if needed

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Target: home win (1) vs not home win (0)
df["y"] = (df["result"] == "H").astype(int)

# Build feature list based on configuration
def get_pca_features(df_columns):
    """
    Dynamically build feature list based on config settings.
    """
    cols_for_pca = []
    
    # Always include odds
    cols_for_pca.extend(["odds_home_win", "odds_draw", "odds_away_win"])
    
    # Add Elo features based on config
    if USE_ELO:
        if USE_DIFF and DELETE_ORIGINAL_DIFF:
            # Only diff
            cols_for_pca.append("elo_diff_pre")
        else:
            # Include available Elo features
            if not USE_DIFF:
                # No diff features, include home/away
                if "elo_home_pre" in df_columns:
                    cols_for_pca.append("elo_home_pre")
                if "elo_away_pre" in df_columns:
                    cols_for_pca.append("elo_away_pre")
            else:
                # Using diff but not deleting originals - include diff
                if "elo_diff_pre" in df_columns:
                    cols_for_pca.append("elo_diff_pre")
    
    # Add form features based on config
    if USE_DIFF and DELETE_ORIGINAL_DIFF:
        # Only diff features
        diff_cols = [c for c in df_columns if c.startswith('form_') and c.endswith('_diff')]
        cols_for_pca.extend(diff_cols)
    else:
        if not USE_DIFF:
            # No diff, include home features (you can customize this)
            home_form_cols = [c for c in df_columns if c.startswith('form_') and c.endswith('_home')]
            cols_for_pca.extend(home_form_cols)
        else:
            # Using diff but keeping originals - include diff
            diff_cols = [c for c in df_columns if c.startswith('form_') and c.endswith('_diff')]
            cols_for_pca.extend(diff_cols)
    
    # Add H2H features based on config
    if USE_H2H:
        if USE_DIFF and DELETE_ORIGINAL_DIFF:
            # Only diff and aggregate H2H features
            h2h_cols = [c for c in df_columns if c.startswith('h2h_') and 
                       (c.endswith('_diff') or c in ['h2h_matches', 'h2h_draws'])]
            cols_for_pca.extend(h2h_cols)
        else:
            if not USE_DIFF:
                # No diff - include basic H2H features
                h2h_cols = ['h2h_matches', 'h2h_draws']
                h2h_cols = [c for c in h2h_cols if c in df_columns]
                cols_for_pca.extend(h2h_cols)
            else:
                # Using diff but keeping originals - include diff
                h2h_diff_cols = [c for c in df_columns if c.startswith('h2h_') and 
                                (c.endswith('_diff') or c in ['h2h_matches', 'h2h_draws'])]
                cols_for_pca.extend(h2h_diff_cols)
    
    # Filter to only columns that exist
    cols_for_pca = [c for c in cols_for_pca if c in df_columns]
    
    return cols_for_pca

cols_for_pca = get_pca_features(df.columns)

print(f"Using {len(cols_for_pca)} features for PCA based on config:")
print(f"  USE_ELO: {USE_ELO}")
print(f"  USE_H2H: {USE_H2H}")
print(f"  USE_DIFF: {USE_DIFF}")
print(f"  DELETE_ORIGINAL_DIFF: {DELETE_ORIGINAL_DIFF}")
print(f"\nFeatures: {cols_for_pca}\n")

X = df[cols_for_pca]
y = df["y"].values

# Safety check: no missing values (if this fails, you need to clean or impute)
assert not X.isnull().any().any(), "NaNs detected in X; clean or add an imputer."


# Cell 3: random stratified split + PCA + logistic regression + CV metrics

# Random stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline: standardize -> PCA -> logistic regression
pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90, random_state=0)),  # keep ~90% variance
        ("log_reg", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ]
)

# Cross-validated performance on full data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
cv_logloss = cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss")

print(f"CV accuracy (mean ± std): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"CV ROC AUC (mean ± std): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"CV log loss (mean ± std): {-cv_logloss.mean():.4f} ± {cv_logloss.std():.4f}")

# Fit on train split and evaluate on held-out test
pipe.fit(X_train, y_train)

y_proba_test = pipe.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= 0.5).astype(int)

print("\nHeld-out test metrics:")
print("Test accuracy:", accuracy_score(y_test, y_pred_test))
print("Test log loss:", log_loss(y_test, pipe.predict_proba(X_test)))
print("Test ROC AUC:", roc_auc_score(y_test, y_proba_test))


# Cell 4: inspect PCA structure and derive feature importance in original space

# Refit on all data to get final PCA + coefficients for interpretation
pipe.fit(X, y)

scaler = pipe.named_steps["scaler"]
pca = pipe.named_steps["pca"]
log_reg = pipe.named_steps["log_reg"]

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

print("Number of PCs used:", pca.n_components_)
print("Explained variance ratio per PC:", explained)
print("Cumulative explained variance:", cum_explained)

# Approximate coefficient for each original (standardized) feature:
# beta_orig ≈ W^T * beta_pcs, where W = pca.components_
if log_reg.coef_.shape[0] == 1:  # binary case
    beta_pcs = log_reg.coef_[0]      # shape (n_components,)
    W = pca.components_              # shape (n_components, n_features)
    beta_orig = beta_pcs @ W         # shape (n_features,)

    feature_importance = pd.Series(beta_orig, index=cols_for_pca)
    feature_importance_abs = feature_importance.abs().sort_values(ascending=False)

    print("\nApprox. feature importance on standardized scale (top 15):")
    print(feature_importance_abs.head(15))
else:
    print("\nMulticlass logistic regression detected.")
    print("Inspect `log_reg.coef_` together with `pca.components_` for class-wise effects.")


# Cell 5: scree plot + PC1 vs PC2 colored by outcome

# Scree plot
components = np.arange(1, len(explained) + 1)

plt.figure()
plt.plot(components, explained, marker="o", label="Individual")
plt.plot(components, cum_explained, marker="o", label="Cumulative")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Scree Plot – PCA")
plt.legend()
plt.show()

# PC1 vs PC2 scatter, colored by outcome (on training set)
X_train_scaled = scaler.transform(X_train)
X_train_pcs = pca.transform(X_train_scaled)

pc1 = X_train_pcs[:, 0]
pc2 = X_train_pcs[:, 1]

plt.figure()
plt.scatter(pc1[y_train == 0], pc2[y_train == 0], alpha=0.4, label="Not Home Win")
plt.scatter(pc1[y_train == 1], pc2[y_train == 1], alpha=0.4, label="Home Win")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PC1 vs PC2 – Colored by Outcome (Train Set)")
plt.legend()
plt.show()


# Cell 6: ROC curve, calibration curve, confusion matrix (test set)

# Recompute probs/preds for clarity
y_proba_test = pipe.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= 0.5).astype(int)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba_test)
auc = roc_auc_score(y_test, y_proba_test)

plt.figure()
plt.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Home Win vs Not (Test Set)")
plt.legend()
plt.show()

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_proba_test, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
plt.xlabel("Predicted probability (binned)")
plt.ylabel("Observed frequency of home win")
plt.title("Calibration Curve – Test Set")
plt.legend()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Not Home Win", "Home Win"])
plt.yticks([0, 1], ["Not Home Win", "Home Win"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.title("Confusion Matrix – Test Set (threshold = 0.5)")
plt.show()


# Cell 7: bar plot for top-N feature importance (absolute |beta|)

N = 15
fi_abs = feature_importance_abs.head(N)

plt.figure()
plt.barh(fi_abs.index[::-1], fi_abs.values[::-1])
plt.xlabel("Absolute importance (|standardized coefficient|)")
plt.title(f"Top {N} Features – Logistic Regression via PCA")
plt.tight_layout()
plt.show()
