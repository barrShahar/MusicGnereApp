import gc

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from GridSearch import perform_grid_search_and_analysis
import GlobalVariables
from sklearn.neighbors import KNeighborsClassifier



def knn_analysis(X_train, y_train, X_test, y_test, extension='', cv=10, sorted_file_names_column=None):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),  # Adding polynomial features
        ('scaler', StandardScaler()),  # Step 1: Scale the data
        ('knn', KNeighborsClassifier())  # Step 2: Train a classifier
    ])

    # Define hyperparameters for the model
    param_grid = {
        'poly__degree': [1, 2, 3],  # Degrees for PolynomialFeatures
        'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 12, 13, 14, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # perform_grid_search_and_analysis(X_train, y_train, X_test, y_test,
    #                                  pipeline,
    #                                  param_grid,
    #                                  GlobalVariables.g_knn_folder,
    #                                  extension,
    #                                  sorted_file_names_column,
    #                                  cv=cv,
    #                                  verbose=2)
