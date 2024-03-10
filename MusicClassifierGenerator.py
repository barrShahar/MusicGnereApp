import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

import GlobalVariables
from Data import Data
from models import knn_analysis, SVM_analysis, adaboost_analysis, perceptron_analysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import os

TRAINED_MODELS_DIR = "trained_models/"
svm_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),  # Adding polynomial features
    ('scaler', StandardScaler()),  # Step 1: Scale the data
    ('classifier', SVC())  # Step 2: Train a classifier
])

# Define hyperparameters for the model
svm_param_grid = {
    'poly__degree': [1],  # Degrees for PolynomialFeatures
    'classifier__C': [1],
    'classifier__gamma': [0.01],
    'classifier__kernel': ['rbf']  # Adding kernel types for exploration
}


# # Define hyperparameters for the model
# svm_param_grid = {
#     'poly__degree': [1, 2],  # Degrees for PolynomialFeatures
#     'classifier__C': [0.1, 1, 10, 100, 10000],
#     'classifier__gamma': [0.001, 0.01, 0.1, 1, 5],
#     'classifier__kernel': ['rbf']  # Adding kernel types for exploration
# }

class MusicClassifierGenerator:
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
        self.extension = TRAINED_MODELS_DIR + str(pairwise_correlation_thresholds) + 'ExclFeat_' + str(feature_count)

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
        print(self.extension)

        if os.path.exists(self.extension + '_model.pkl'):
            self.load_model(self.extension)
            print("model exist and has been loaded")
            return

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

    def predict(self, x):

        s = f"Genre: {self.best_classifier.predict(x)}\n"
        calibrated_svm = CalibratedClassifierCV(self.best_classifier, method='sigmoid', cv='prefit')
        calibrated_svm.fit(self.X_train, self.y_train)
        # Predict probabilities
        probabilities = calibrated_svm.predict_proba(x)
        # Assuming 'classes' is a list of class labels
        for i, class_label in enumerate(calibrated_svm.classes_):
            class_probability = probabilities[0][i]
            s += f"Probability of {class_label}: {class_probability:.3f}\n"
            print(f"Probability of {class_label}: {class_probability:.3f}")
        return s

    def predict_given_all_features(self, x: pd.DataFrame):
        filtered_x = x[self.feature_names_list]
        return self.predict(filtered_x)

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

    def statistics(self):

        y_pred = self.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("conf_matrix:\n", conf_matrix)

        # Display and save the confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.best_classifier.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix ' + "{:.3f}".format(accuracy))
        # plt.savefig(output_folder + file_suffix + ' ' + str(accuracy) + str(
        #     grid.best_params_) + f"confusion_matrix_{timestamp}.png")
        plt.show()

    def set_default_classifier(self):
        self.set_features_params(56, 1)
        self.train_model_via_grid_search(svm_pipeline, svm_param_grid, cv=10, verbose=False)


if __name__ == '__main__':
    music_app = MusicClassifierGenerator()
    music_app.set_features_params(56, 1)
    music_app.train_model_via_grid_search(svm_pipeline, svm_param_grid, cv=10, verbose=True)
    ext = music_app.save_model()
    y_pred1 = music_app.predict(music_app.X_test)
    music_app.statistics()
