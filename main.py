from data_preprocessing import load_and_preprocess_data
from pca_reduction import apply_pca
from train_models import train_svm_linear, train_svm_rbf, train_random_forest
from train_mlp import create_mlp_classifier  # Import the new function
from evaluate_models import evaluate_model, plot_confusion_matrix, confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

def main():
    # Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, target_mapping = load_and_preprocess_data("updated_pollution_dataset.csv")

    # Apply PCA with progress bar for SVM and Random Forest
    print("Applying PCA...")
    with tqdm(total=1, desc="PCA Reduction") as pbar:
        X_train_pca, X_test_pca, explained_variance = apply_pca(X_train, X_test, n_components=5)
        pbar.update(1)
    print("Explained Variance Ratio:", np.round(explained_variance, 3))

    # Train models with progress bar
    print("Training models...")
    with tqdm(total=4, desc="Model Training") as pbar:
        best_svm_linear, y_pred_linear = train_svm_linear(X_train_pca, y_train, X_test_pca)
        pbar.update(1)

        best_svm_rbf, y_pred_rbf = train_svm_rbf(X_train_pca, y_train, X_test_pca)
        pbar.update(1)

        best_rf, y_pred_rf = train_random_forest(X_train_pca, y_train, X_test_pca)
        pbar.update(1)

        # Create and train MLP model with GridSearchCV
        mlp_model = create_mlp_classifier(X_train, len(set(y_train)))  # Pass the number of classes
        param_grid = {
            'batch_size': [16, 32], 
            'epochs': [50, 100],
            'learning_rate': [0.001, 0.01]  # Optional, can add more hyperparameters
        }

        grid_search = GridSearchCV(estimator=mlp_model, param_grid=param_grid, cv=3, verbose=10)
        grid_search.fit(X_train, y_train)

        # Get the best model after GridSearchCV
        best_mlp_model = grid_search.best_estimator_

        # Predict using the best model
        y_pred_mlp = np.argmax(best_mlp_model.predict(X_test), axis=1)

        pbar.update(1)

    # Evaluate models
    print("SVM Linear Kernel Results:")
    evaluate_model(best_svm_linear, X_train_pca, X_test_pca, y_train, y_test)

    print("\nSVM RBF Kernel Results:")
    evaluate_model(best_svm_rbf, X_train_pca, X_test_pca, y_train, y_test)

    print("\nRandom Forest Results:")
    evaluate_model(best_rf, X_train_pca, X_test_pca, y_train, y_test)

    print("\nMLP Results:")
    evaluate_model(best_mlp_model, X_train, X_test, y_train, y_test)

    # Plot confusion matrices with progress bar
    print("Plotting confusion matrices...")
    with tqdm(total=4, desc="Plotting") as pbar:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred_linear), "SVM Linear Kernel")
        pbar.update(1)

        plot_confusion_matrix(confusion_matrix(y_test, y_pred_rbf), "SVM RBF Kernel")
        pbar.update(1)

        plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf), "Random Forest")
        pbar.update(1)

        plot_confusion_matrix(confusion_matrix(y_test, y_pred_mlp), "MLP")
        pbar.update(1)

if __name__ == "__main__":
    main()