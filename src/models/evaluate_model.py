"""
Evaluate our model using metrics (accuracy, f1, etc.), confusion matrices, etc.
"""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate(model, X_test, y_test):
    # Predict on holdout set.
    y_pred  = model.predict(X_test)

    # Print basic statistics.
    print(f"Accuracy:          {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification report (0=H, 1=A, 2=D):")
    print(classification_report(y_test, y_pred, digits=3))
    
    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
    ConfusionMatrixDisplay(cm, display_labels=['Home','Away','Draw']).plot(values_format='d')