from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import GlobalVariables

# from GridSearch import perform_grid_search_and_analysis


def perceptron_analysis(X_train, y_train, X_test, y_test, extension='', cv=10):
    # Define the pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),  # Adding polynomial features
        ('scaler', StandardScaler()),  # Step 1: Scale the data
        ('perceptron', Perceptron())
    ])

    # Define the parameter grid
    param_grid = {
        'poly__degree': [1, 2, 3, 4],  # Degrees for PolynomialFeatures
        'perceptron__max_iter': [400, 500, 1000, 2000],
        'perceptron__eta0': [0.1, 0.01, 0.001, 0.0001, 0.00001],  # Learning rate
        'perceptron__tol': [1e-4, 1e-5, 1e-6],
        'perceptron__penalty': [None, 'l2', 'l1', 'elasticnet'],
        'perceptron__alpha': [0.00001, 0.0001, 0.001]
    }

    # # Call the grid_search_func
    # perform_grid_search_and_analysis(X_train, y_train, X_test, y_test, pipeline,
    #                                  param_grid, GlobalVariables.g_perceptron, extension, cv=cv, verbose=2)
