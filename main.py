from data_preprocessing import load_and_preprocess_data
from pca_reduction import apply_pca
from train_models import train_svm_linear, train_svm_rbf, train_random_forest
from evaluate_models import evaluate_model, plot_confusion_matrix, confusion_matrix
import numpy as np
from tqdm import tqdm

def main():
    # Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, target_mapping = load_and_preprocess_data("updated_pollution_dataset.csv")

    # Apply PCA with progress bar
    print("Applying PCA...")
    with tqdm(total=1, desc="PCA Reduction") as pbar:
        X_train_pca, X_test_pca, explained_variance = apply_pca(X_train, X_test, n_components=5)
        pbar.update(1)
    print("Explained Variance Ratio:", np.round(explained_variance, 3))

    # Train models with progress bar
    print("Training models...")
    with tqdm(total=3, desc="Model Training") as pbar:
        best_svm_linear, y_pred_linear = train_svm_linear(X_train_pca, y_train, X_test_pca)
        pbar.update(1)

        best_svm_rbf, y_pred_rbf = train_svm_rbf(X_train_pca, y_train, X_test_pca)
        pbar.update(1)

        best_rf, y_pred_rf = train_random_forest(X_train_pca, y_train, X_test_pca)
        pbar.update(1)

    # Evaluate models
    print("SVM Linear Kernel Results:")
    evaluate_model(best_svm_linear, X_train_pca, X_test_pca, y_train, y_test)

    print("\nSVM RBF Kernel Results:")
    evaluate_model(best_svm_rbf, X_train_pca, X_test_pca, y_train, y_test)

    print("\nRandom Forest Results:")
    evaluate_model(best_rf, X_train_pca, X_test_pca, y_train, y_test)

    # Plot confusion matrices with progress bar
    print("Plotting confusion matrices...")
    with tqdm(total=3, desc="Plotting") as pbar:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred_linear), "SVM Linear Kernel")
        pbar.update(1)

        plot_confusion_matrix(confusion_matrix(y_test, y_pred_rbf), "SVM RBF Kernel")
        pbar.update(1)

        plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf), "Random Forest")
        pbar.update(1)

if __name__ == "__main__":
    main()