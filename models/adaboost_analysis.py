from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import GlobalVariables
# from GridSearch import perform_grid_search_and_analysis


def adaboost_analysis(X_train, y_train, X_test, y_test, extension='', cv=10):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('pca', PCA()),
        ('adaboost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)))
    ])

    # 2. Define the grid of hyperparameters to search
    param_grid1 = {
        'poly__degree': [1, 2, 3],
        'adaboost__n_estimators': [50, 100, 150, 200, 215],
        'adaboost__learning_rate': [1.0],
        'adaboost__estimator__max_depth': [1, 2, 3, 4, 5, 6, 7],  # tuning the depth of the decision tree in AdaBoost
    }

    param_grid2 = {
        'poly__degree': [1, 2, 3],
        'pca': [PCA()],  # Here's where we specify to either use PCA or not
        'pca__n_components': [0.95, 0.90, 0.85],
        'adaboost__n_estimators': [50, 100, 150, 200],
        'adaboost__learning_rate': [1.0],
        'adaboost__estimator__max_depth': [1, 2, 3, 4, 5, 6, 7],  # tuning the depth of the decision tree in AdaBoost
    }

    # combined_grid = [param_grid1, param_grid2]

    # perform_grid_search_and_analysis(X_train, y_train, X_test, y_test,
    #                                  pipeline, param_grid1, GlobalVariables.g_adaboost_folder,
    #                                  extension, cv=cv, verbose=2)
