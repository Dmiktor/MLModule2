from sklearn.decomposition import PCA

def apply_pca(X_train, X_test, n_components=5):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca.explained_variance_ratio_