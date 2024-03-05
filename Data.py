from typing import Tuple, List, Any

import pandas as pd
from sklearn.model_selection import train_test_split

import GlobalVariables
import matplotlib.pyplot as plt


def fetch_dataset3(path, get_files_name=False) -> pd.DataFrame:
    # Load main data
    data = pd.read_csv(path)
    if not get_files_name:
        data.drop('filename', axis=1, inplace=True)

    # removing indexing list
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', axis=1, inplace=True)

    return data[data['label'].isin(GlobalVariables.selected_genre)]


def extract_and_remove_column(dataframes: list, column_name: str) -> list:
    """
    Extracts a column from the first DataFrame in a list and then removes that column from all DataFrames in the list.

    Parameters:
    - dataframes (list of pd.DataFrame): A list of pandas DataFrames from which the specified column should be removed.
    - column_name (str): The name of the column to extract from the first DataFrame and remove from all DataFrames.

    Returns:
    - pd.Series: The extracted column from the first DataFrame in the list.
    """

    # Extracting the desired column from the first DataFrame
    extracted_column_list = []

    # Iterating over all DataFrames in the list and removing the specified column
    for dataframe in dataframes:
        extracted_column_list.append(dataframe[column_name])
        dataframe.drop(column_name, axis=1, inplace=True)

    return extracted_column_list


def get_data(path):
    dataset = fetch_dataset3(path=path, get_files_name=True)
    file_names_column = extract_and_remove_column([dataset], 'filename')[0]

    return dataset, file_names_column


def convert_labels_to_numerical(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical labels in the 'label' column to numerical values.

    Parameters:
    - DataFrame (DataFrame): Input data containing a 'label' column.

    Returns:
    - DataFrame: The input dataframe with categorical labels in the 'label' column replaced by numerical values.
    """

    # Retrieve unique labels and create a mapping from labels to numbers
    unique_labels = dataframe['label'].unique()
    label_to_numeric_mapping = {label: index for index, label in enumerate(unique_labels)}

    # Apply the mapping to convert the 'label' column to numerical values
    dataframe.loc[:, 'label'] = dataframe['label'].map(label_to_numeric_mapping)

    return dataframe


def visualize_top_correlated_feature_pairs(sorted_correlations: pd.Series) -> None:
    """
    Plot the top feature pairs with the highest absolute correlation values.

    Parameters:
    - sorted_correlations (Series): A sorted series of feature pair correlations.

    Returns:
    - None
    """

    # Remove every second element to avoid duplicates (since corr(X,Y) = corr(Y,X))
    sorted_correlations = sorted_correlations.iloc[::2]

    # Extract top correlated feature pairs (top 40 pairs)
    top_pairs = sorted_correlations[:40].copy()

    # Reverse order for better visualization in horizontal bar plot
    plotting_pairs = top_pairs.iloc[::-1]

    # Plotting the data
    plt.figure(figsize=(10, 5))
    plotting_pairs.plot(kind='barh')
    plt.title('Top Correlated Pairs')
    plt.ylabel('Absolute Correlation')
    plt.tight_layout()
    plt.show()


def compute_pairwise_correlations(features: pd.DataFrame) -> pd.Series:
    """
    Calculate pairwise correlations between features.

    Parameters:
    - feature_data (DataFrame): Input data containing only the features.

    Returns:
    - Series: Sorted pairwise feature correlations.
    """

    # Compute the absolute correlation matrix
    corr_matrix = features.corr().abs()

    # Unstack the matrix to get pairs and then sort them
    correlations = corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)

    # Exclude self-correlations, which will always be 1 (diagonal of the matrix)
    correlations = correlations[correlations < 1]

    # If plotting is enabled (controlled by global variable 'g_plot')
    if GlobalVariables.g_plot:
        visualize_top_correlated_feature_pairs(correlations.iloc[::2])

    return correlations


def visualize_correlation_with_target(sorted_correlation: pd.Series) -> None:
    """
    Display a bar chart of features' correlation with the target variable.

    :param sorted_correlation: Sorted correlation values.
    """
    plt.figure(figsize=(12, 8))
    sorted_correlation.plot(kind='bar')
    plt.title('Feature Correlation with Target Variable')
    plt.ylabel('Absolute Correlation')
    plt.xlabel('Features')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def select_important_features(data: pd.DataFrame, max_features: int, correlation_threshold: float = 0.8) -> (list, int):
    """
    Select the most important features from the provided data, based on their correlation
    with the target label and inter-feature correlation.

    Parameters:
    - data (DataFrame): The dataset containing features and target labels.
    - max_features (int): The maximum number of features to select.
    - correlation_threshold (float, optional): The threshold for inter-feature correlation above which
                                         features will be considered redundant. Default is 0.8.

    Returns:
    - list: A list of selected feature names.
    - int: The count of selected features.
    """

    # Make a copy to prevent modifying the original data
    copied_data = data.copy()

    # Convert text labels into numbers for correlation calculations
    copied_data = convert_labels_to_numerical(copied_data)
    X = copied_data.drop('label', axis=1)
    y = copied_data['label']

    # Calculate correlations between features and the target label
    correlation_series = X.corrwith(y).abs()
    sorted_correlation = correlation_series.sort_values(ascending=False)

    print(sorted_correlation)

    # plot if global variable g_plot is set
    if GlobalVariables.g_plot:
        visualize_correlation_with_target(sorted_correlation)

    # Calculate pairwise feature correlations
    pairwise_feature_correlations = compute_pairwise_correlations(X)

    significant_features = []
    while len(significant_features) < max_features and not sorted_correlation.empty:
        # Get the feature with the highest correlation to the target
        top_feature = sorted_correlation.idxmax()
        significant_features.append(top_feature)

        # Identify features that are highly correlated with the best feature
        redundant_features = pairwise_feature_correlations[top_feature][
            pairwise_feature_correlations[top_feature] > correlation_threshold].index.tolist()

        # Remove best feature and highly correlated features from further consideration
        features_to_exclude = [top_feature] + redundant_features
        sorted_correlation = sorted_correlation.drop(features_to_exclude, errors='ignore')

    print("The most significant_features:\n", significant_features)
    print("Number of features: ", len(significant_features))
    # For plotting purposes
    if GlobalVariables.g_plot:
        visualize_correlation_with_target(correlation_series[significant_features])
    return significant_features, len(significant_features)


class Data:
    def __init__(self, path=GlobalVariables.dataset_path):
        self.dataset, self.file_names_column = get_data(path=path)
        self.y = self.dataset['label']

    def get_feature_names_list(self, max_feature_num: int, pairwise_correlation_thresholds: float) -> tuple[list[Any], int]:
        important_features, feature_count = select_important_features \
            (data=self.dataset,
             max_features=max_feature_num,  # Set the upper limit for the number of features to select
             correlation_threshold=pairwise_correlation_thresholds)  # Max correlation between two features

        return important_features, feature_count

    def get_features_data(self, feature_names_list: list):
        return self.dataset[feature_names_list]

    def train_test_split(self, X, y):
        tmp_X = X.copy()
        tmp_X['filename'] = self.file_names_column

        # Split data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(tmp_X, y, test_size=0.2, random_state=0, stratify=y)

        # Extract the 'filename' column from both X_train and X_test datasets, then remove it from both.
        sorted_file_names_column = extract_and_remove_column([X_train, X_test], 'filename')

        del tmp_X
        return X_train, X_test, y_train, y_test, sorted_file_names_column

