from sklearn.model_selection import GridSearchCV
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Example hyperparameters for demonstration
    param_grid = {'C': [0.1, 1, 10]} if 'SVC' in str(model) else {'n_estimators': [50, 100, 150]}

    print("Performing GridSearchCV...")

    total_steps = len(param_grid) * 5  # 5 is for cv=5 folds
    with tqdm(total=total_steps, desc="Grid Search Progress") as pbar:
        grid_search = GridSearchCV(model, param_grid, cv=5, verbose=10)
        grid_search.fit(X_train, y_train)
        pbar.update(total_steps)  

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate and display accuracy and confusion matrix
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    plot_confusion_matrix(cm, model.__class__.__name__)

def plot_confusion_matrix(cm, model_name):
    # Create Results directory if it doesn't exist
    if not os.path.exists('Results'):
        os.makedirs('Results')

    # Create the plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')

    # Save the plot to the "Results" folder
    file_path = os.path.join('Results', f'{model_name}_confusion_matrix.png')
    plt.savefig(file_path)
    print(f"Confusion Matrix saved to {file_path}")
    
    # Optionally, you can also display the plot
    plt.show()