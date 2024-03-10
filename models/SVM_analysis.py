from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import GlobalVariables

# from GridSearch import perform_grid_search_and_analysis



def svm_analysis(X_train, y_train, X_test, y_test, extension='', cv=10, sorted_file_names_column=None):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),  # Adding polynomial features
        ('scaler', StandardScaler()),  # Step 1: Scale the data
        ('classifier', SVC())  # Step 2: Train a classifier
    ])

    # Define hyperparameters for the model
    param_grid = {
        'poly__degree': [1, 2],  # Degrees for PolynomialFeatures
        'classifier__C': [0.1, 1, 10, 100, 10000],
        'classifier__gamma': [0.001, 0.01, 0.1, 1, 5],
        'classifier__kernel': ['rbf']  # Adding kernel types for exploration
    }

    # return perform_grid_search_and_analysis(X_train, y_train, X_test, y_test,
    #                                         pipeline,
    #                                         param_grid,
    #                                         GlobalVariables.g_SVM_folder,
    #                                         extension,
    #                                         sorted_file_names_column,
    #                                         cv=cv,
    #                                         verbose=2)
