import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import GlobalVariables
from Data import Data
from models import knn_analysis, SVM_analysis, adaboost_analysis, perceptron_analysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import os

svm_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),  # Adding polynomial features
    ('scaler', StandardScaler()),  # Step 1: Scale the data
    ('classifier', SVC())  # Step 2: Train a classifier
])

# Define hyperparameters for the model
svm_param_grid = {
    'poly__degree': [1],  # Degrees for PolynomialFeatures
    'classifier__C': [100],
    'classifier__gamma': [0.1],
    'classifier__kernel': ['rbf']  # Adding kernel types for exploration
}


# # Define hyperparameters for the model
# svm_param_grid = {
#     'poly__degree': [1, 2],  # Degrees for PolynomialFeatures
#     'classifier__C': [0.1, 1, 10, 100, 10000],
#     'classifier__gamma': [0.001, 0.01, 0.1, 1, 5],
#     'classifier__kernel': ['rbf']  # Adding kernel types for exploration
# }

class MusicApplication:
    def __init__(self):
        self.file_suffix = None
        self.param_grid = None
        self.pipeline = None
        self.best_classifier = None
        self.feature_names_list = None
        self.extension = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sorted_file_names_column = None
        self.X = None
        self.data = Data(GlobalVariables.dataset_path)

    def set_features_params(self, max_feature_num: int, pairwise_correlation_thresholds: float):
        self.feature_names_list, feature_count = self.data.get_feature_names_list(max_feature_num,
                                                                                  pairwise_correlation_thresholds)
        self.X = self.data.get_features_data(self.feature_names_list)

        # Split data into training and testing sets (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test, self.sorted_file_names_column = self.data.train_test_split(
            self.X, self.data.y)

        # Generate a naming extension for file naming purposes.
        self.extension = str(pairwise_correlation_thresholds) + 'ExclFeat_' + str(feature_count)

    def train_model(self, max_feature_num: int, pairwise_correlation_thresholds: float):
        pass

    def train_model_via_grid_search(self, pipeline, param_grid, cv=10, verbose=False):

        self.file_suffix = str(pipeline) + str(param_grid)
        self.extension = self.extension + self.file_suffix

        if os.path.exists(self.extension + '_model.pkl'):
            self.load_model(self.extension)
            print("model exist and has been loaded")
            return

        # Conduct the grid search
        grid = GridSearchCV(pipeline, param_grid, cv=cv, verbose=verbose, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        print(str(pipeline))
        print(str(param_grid))

        # TODO print(file_suffix + " Best parameters:", grid.best_params_)
        print("Best cross-validation score:", grid.best_score_)

        # Train the best model on the training data
        self.best_classifier = grid.best_estimator_
        self.best_classifier.fit(self.X_train, self.y_train)
        self.pipeline = pipeline
        self.param_grid = param_grid

    def save_model(self):
        # Save the trained model
        joblib.dump(self.best_classifier, self.extension + '_model.pkl')

        # Save feature names list
        with open(self.extension + '_features.txt', 'w') as f:
            f.write('\n'.join(self.feature_names_list))

        # Save other necessary data
        joblib.dump({
            'extension': self.extension,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'sorted_file_names_column': self.sorted_file_names_column,
            'param_grid': self.param_grid,
            'pipeline': self.pipeline
        }, self.extension + '_data.pkl')

        return self.extension

    def predict(self, samples):
        return self.best_classifier.predict(samples)

    def load_model(self, file_path):
        # Load the trained model
        self.best_classifier = joblib.load(file_path + '_model.pkl')

        # Load feature names list
        with open(file_path + '_features.txt', 'r') as f:
            self.feature_names_list = f.read().splitlines()

        # Load other necessary data
        data = joblib.load(file_path + '_data.pkl')
        self.extension = data['extension']
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.sorted_file_names_column = data['sorted_file_names_column']
        self.param_grid = data['param_grid']
        self.pipeline = data['pipeline']

    def shazam(self):
        pass


if __name__ == '__main__':
    music_app = MusicApplication()
    music_app.set_features_params(48, 1)
    music_app.train_model_via_grid_search(svm_pipeline, svm_param_grid, cv=10, verbose=True)
    ext = music_app.save_model()
    y_pred1 = music_app.predict(music_app.X_test)

    lMusic_app = MusicApplication()
    lMusic_app.load_model(ext)
    y_pred2 = lMusic_app.predict(lMusic_app.X_test)

    if np.array_equal(y_pred1, y_pred2):
        print(True)
    else:
        print(False)
