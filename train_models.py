from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Train SVM with a linear kernel (without GridSearch for quick testing)
def train_svm_linear(X_train, y_train, X_test):
    svc_linear = SVC(kernel='linear', C=1)  # Default C=1 without GridSearch
    svc_linear.fit(X_train, y_train)
    y_pred = svc_linear.predict(X_test)
    return svc_linear, y_pred

# Train SVM with an RBF kernel (without GridSearch for quick testing)
def train_svm_rbf(X_train, y_train, X_test):
    svc_rbf = SVC(kernel='rbf', C=1, gamma='scale')  # Default C=1, gamma='scale' without GridSearch
    svc_rbf.fit(X_train, y_train)
    y_pred = svc_rbf.predict(X_test)
    return svc_rbf, y_pred

# Train Random Forest with parallelism enabled and no GridSearch (quick testing)
def train_random_forest(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)  # Use all cores
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_pred

# If you want to keep GridSearch with parallelism, here is an example for Random Forest
def train_random_forest_gridsearch(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_jobs=-1)  # Use all cores for faster training
    param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, n_jobs=-1)  # Parallel GridSearch
    grid_search_rf.fit(X_train, y_train)
    best_model = grid_search_rf.best_estimator_
    y_pred = best_model.predict(X_test)
    return best_model, y_pred

# Train SVM Linear with GridSearch and parallelism (if required)
def train_svm_linear_gridsearch(X_train, y_train, X_test):
    svc_linear = SVC(kernel='linear')
    param_grid_linear = {'C': [0.1, 1, 10]}
    grid_search_linear = GridSearchCV(svc_linear, param_grid_linear, cv=5, n_jobs=-1)  # Parallel GridSearch
    grid_search_linear.fit(X_train, y_train)
    best_model = grid_search_linear.best_estimator_
    y_pred = best_model.predict(X_test)
    return best_model, y_pred

# Train SVM RBF with GridSearch and parallelism (if required)
def train_svm_rbf_gridsearch(X_train, y_train, X_test):
    svc_rbf = SVC(kernel='rbf')
    param_grid_rbf = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    grid_search_rbf = GridSearchCV(svc_rbf, param_grid_rbf, cv=5, n_jobs=-1)  # Parallel GridSearch
    grid_search_rbf.fit(X_train, y_train)
    best_model = grid_search_rbf.best_estimator_
    y_pred = best_model.predict(X_test)
    return best_model, y_pred