import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Loads the dataset from the specified CSV file path.
    """
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data):
    """
    Fills missing values in the dataset.
    numerical columns are filled with the mean and categorical columns with the mode.
    """
    for column in data.columns:
        if data[column].dtype == 'object':  # Categorical columns
            # Avoid chaining and reassign the column to the DataFrame
            data[column] = data[column].fillna(data[column].mode()[0])
        else:  # Numerical columns
            # Avoid chaining and reassign the column to the DataFrame
            data[column] = data[column].fillna(data[column].mean())
    return data


def encode_categorical_variables(data):
    """
    encodes categorical variables using one-hot encoding.
    """
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    return data

def normalize_features(data):
    """
    normalizes numerical features using Z-score normalization (standardization).
    """
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].mean()) / data[numerical_columns].std()
    return data

def preprocess_data(file_path):
    """
    loads, handles missing values, encodes categorical variables, and normalizes the dataset.
    """
    # Step 1: Load the data
    data = load_data(file_path)
    print("Data loaded successfully.")
    
    # Step 2: Handle missing values
    data = handle_missing_values(data)
    print("Missing values handled.")
    
    # Step 3: Encode categorical variables
    data = encode_categorical_variables(data)
    print("Categorical variables encoded.")
    
    # Step 4: Normalize numerical features
    data = normalize_features(data)
    print("Features normalized.")
    
    return data

if __name__ == '__main__':
    # file path to the regression dataset
    file_path = 'regression_task/data/fuel_train.csv'
    
    # Preprocessing the data
    processed_data = preprocess_data(file_path)
    
    # Save the preprocessed data
    processed_data.to_csv('regression_task/data/preprocessed_training_data.csv', index=False)
    print("Preprocessed data saved.")
