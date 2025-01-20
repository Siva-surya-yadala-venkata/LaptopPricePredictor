


import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import prepare_data
from model import RobustMLP
from utils import stratified_split, calculate_metrics, print_evaluation_results

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    # Load and validate data
    try:
        data = pd.read_csv("laptop_data.csv")
        print("Available columns:", list(data.columns))
    except FileNotFoundError:
        print("Error: laptop_data.csv not found!")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Filter only numerical columns for the correlation matrix
    numerical_data = data.select_dtypes(include=[np.number])

    # Compute and visualize the correlation matrix
    correlation_matrix = numerical_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix (Numerical Features)")
    plt.show()

    # Save the correlation matrix to a CSV file
    correlation_matrix.to_csv("correlation_matrix.csv")
    print("Correlation matrix saved to correlation_matrix.csv.")

    # Prepare the data
    try:
        X, y = prepare_data(data)
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return

    # Split data
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train, X_val, y_train, y_val = stratified_split(X_train, y_train)

    # Normalize target variable
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    y_train_scaled = (y_train - y_train_mean) / y_train_std
    y_val_scaled = (y_val - y_train_mean) / y_train_std
    y_test_scaled = (y_test - y_train_mean) / y_train_std

    # Initialize and train model with improved hyperparameters
    try:
        mlp_model = RobustMLP(
            input_size=X_train.shape[1], 
            hidden_layers=[512, 256, 128],  # Deeper network
            output_size=1, 
            learning_rate=0.0002,  # Lower learning rate for stability
            dropout_rate=0.3,
            l2_lambda=0.00001,  # Reduced regularization
            momentum=0.95
        )

        history = mlp_model.train(
            X_train, y_train_scaled, 
            X_val, y_val_scaled, 
            epochs=2000,  # More epochs
            batch_size=32,  # Smaller batch size
            early_stopping_patience=100  # More patience
        )
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return

    # Predictions and evaluation
    y_pred_scaled = mlp_model.predict(X_test)
    y_pred = y_pred_scaled * y_train_std + y_train_mean

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred.flatten())

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Actual Price': y_test,
        'Predicted Price': y_pred.flatten(),
        'Absolute Error': np.abs(y_test - y_pred.flatten()),
        'Percentage Error': np.abs((y_test - y_pred.flatten()) / y_test) * 100
    })

    # Save only actual and predicted prices to a CSV file
    output_file = "actual_vs_predicted_prices.csv"
    comparison_df[['Actual Price', 'Predicted Price']].to_csv(output_file, index=False)
    print(f"\nAll actual and predicted prices saved to {output_file}.")

    # Print evaluation results
    print_evaluation_results(metrics=metrics, comparison_df=comparison_df)

    # Print top 10 predictions in the console
    print("\nTop 10 Predicted vs Actual Prices:")
    print(comparison_df.head(10).to_string(index=False))

    # Analyze and save high-error rows
    high_error_threshold = comparison_df['Percentage Error'].quantile(0.75)
    high_error_cases = comparison_df[comparison_df['Percentage Error'] >= high_error_threshold]
    high_error_cases.to_csv("high_error_cases.csv", index=False)
    print(f"\nHigh-error cases (≥75th percentile) saved to high_error_cases.csv.")

    # Visualize predictions vs actual values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred.flatten(), alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize percentage error distribution with percentiles
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    percentile_values = comparison_df['Percentage Error'].quantile(percentiles)
    plt.figure(figsize=(10, 6))
    sns.histplot(comparison_df['Percentage Error'], kde=True, color='blue', bins=30)
    for p, value in zip(percentiles, percentile_values):
        plt.axvline(value, color='red', linestyle='--', label=f'{int(p*100)}th Percentile: {value:.2f}%')
    plt.title("Percentage Error Distribution with Percentiles")
    plt.xlabel("Percentage Error (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
