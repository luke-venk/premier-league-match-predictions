"""
Evaluate our model using metrics (accuracy, f1, etc.), confusion matrices, etc.
"""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.config import N_MATCHES, MODEL
import matplotlib.pyplot as plt

def evaluate(model, X_test, y_test, show_confusion_matrix=False):
    # Predict on holdout set.
    y_pred = model.predict(X_test)

    print('-' * 50)
    print(f'Model type = {MODEL.name}')
    print(f'N_MATCHES = {N_MATCHES}')
    
    # Print basic statistics.
    print(f'\nAccuracy = {accuracy_score(y_test, y_pred):.3f}')
    
    # Classification report.
    print('\nClassification report (0 = Home Win, 1 = Draw, 2 = Away Win):')
    print(classification_report(y_test, y_pred, digits=3))
    print('-' * 50)
    
    # Confusion matrix.
    if show_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Home Win', 'Draw', 'Away Win'])
        disp.plot(values_format='d')
        plt.show()