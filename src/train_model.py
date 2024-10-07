import numpy as np
import pandas as pd
import pickle  # Import the pickle module

class LinearRegressionFromScratch:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias (intercept) term
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coefficients)

def train_model(file_path):
    # Load the preprocessed data
    data = pd.read_csv(file_path)
    
    # Drop unnecessary columns (categorical columns and identifiers)
    data = data.drop(columns=['Year', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'TRANSMISSION', 'FUEL'])
    
    # Set the correct target column ('FUEL CONSUMPTION')
    target_column = 'FUEL CONSUMPTION'
    
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Initialize the linear regression model
    model = LinearRegressionFromScratch()
    
    # Train the model
    model.fit(X.values, y.values)
        
    # Save the entire model using pickle
    with open('regression_task/models/regression_model_final.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved successfully using pickle.")
    
    return model

if __name__ == '__main__':
    file_path = 'regression_task/data/fuel_train.csv'
    model = train_model(file_path)
