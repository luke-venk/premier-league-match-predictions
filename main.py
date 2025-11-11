"""
Entry point for our program. This runs the entire pipeline, from loading the data, feature engineering,
training the model, and evaluation.
"""
import os
import pandas as pd
from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, N_MATCHES, SPORTSBOOK
from src.data.load_data import load_matches
from src.data.build_features import build_rolling_features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

def main():
    # Debug flag to always redo dataset until proven cleaning works.
    # TODO: remove
    RESET = True
    
    # If the processed data for the configured season has already been loaded, reuse it.
    if os.path.exists(PROCESSED_DATA_PATH) and not RESET:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    # Otherwise, load and process the dataset, saving it to the processed data directory for future use.
    else:
        df_raw = load_matches(csv_path=RAW_DATA_PATH, sportsbook=SPORTSBOOK)
        df = build_rolling_features(df=df_raw, n_matches=N_MATCHES)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(df.head())

    # map from results to softmax output
    label_map = {'H':0, 'A':1, 'D':2}

    #copy df to save original
    df_proc = df.copy()

    #use only previous data + odds
    feature_cols = [c for c in df_proc.columns if c.startswith('form_')]
    feature_cols += ['odds_home_win', 'odds_draw', 'odds_away_win']

    #make feature matrix X and solutions y
    X = df_proc[feature_cols].copy()
    y = df_proc['result'].map(label_map).astype(int)

    #chat said i should sort it one last time just in case
    df_proc = df_proc.sort_values('date').reset_index(drop=True)
    #update X and y to sorted indexs
    X = X.loc[df_proc.index].reset_index(drop=True)
    y = y.loc[df_proc.index].reset_index(drop=True).values

    #70-30 training split
    split = .7
    cut = int(split * len(df_proc))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y[:cut], y[cut:]

    #make pipeline no regularization Most basic model
    pipe = Pipeline([
        ('scaler', StandardScaler()), #centers/scales features so they’re comparable. recommened by chat
        ('clf', LogisticRegression(     #standard logistic regression
        multi_class='multinomial', solver='lbfgs', #quasi newton method good for multinomial on medium tabular data
        max_iter=5000,
        ))
    ])  

    #fit
    pipe.fit(X_train, y_train)

    #predict
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    #print some basic stats
    print(f"Accuracy:          {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification report (0=H, 1=A, 2=D):")
    print(classification_report(y_test, y_pred, digits=3))
    
    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
    ConfusionMatrixDisplay(cm, display_labels=['Home','Away','Draw']).plot(values_format='d')


   
        

if __name__ == "__main__":
    main()