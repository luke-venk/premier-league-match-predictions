from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_model(X_train, y_train):
    # Logistic regression with no regularization: most basic model.
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Data standardization.
        ('clf', LogisticRegression(     # Classifier is Logistic Regression.
            multi_class='multinomial',  # Quasi newton method good for multinomial on medium tabular data
            solver='lbfgs',
            max_iter=5000
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline