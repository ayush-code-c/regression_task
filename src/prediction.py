import numpy as np
import pandas as pd
import pickle  # Import pickle to load the model

class LinearRegressionFromScratch:
    def _init_(self):
        self.coefficients = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias (intercept) term
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coefficients)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def evaluate_and_save_predictions(file_path):
    # Load the preprocessed data
    data = pd.read_csv(file_path)
    
    # Drop unnecessary columns (same as in training)
    data = data.drop(columns=['Year', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'TRANSMISSION', 'FUEL'])
    
    # Set the correct target column
    target_column = 'FUEL CONSUMPTION'
    
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Load the pickled model
    with open('regression_task/models/regression_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Predict the target values using the loaded model
    y_pred = model.predict(X.values)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    r2 = r_squared(y, y_pred)
    
    # Print the evaluation metrics in the output terminal
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Save the evaluation metrics to a file
    with open('regression_task/results/train_metrics.txt', 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error: {mse:.2f}\n")
        f.write(f"Root Mean Squared Error: {rmse:.2f}\n")
        f.write(f"R-squared: {r2:.2f}\n")
    
    print("Evaluation metrics saved successfully.")
    
    # Save the predictions to a CSV file without a header as per the instructions
    np.savetxt('regression_task/results/train_predictions.csv', y_pred, delimiter=',', fmt='%f', header='', comments='')
    
    print("Predictions saved successfully in train_predictions.csv.")

if __name__ == '__main__':
    file_path = 'regression_task/data/fuel_train.csv'
    evaluate_and_save_predictions(file_path)