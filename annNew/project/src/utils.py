import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(X, y, test_size=0.2, random_state=42):
    """Perform stratified split of data based on price ranges."""
    # Create price bins for stratification
    y_bins = pd.qcut(y, q=5, labels=False)
    
    # Use sklearn's train_test_split with stratification
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, 
                          stratify=y_bins)

def calculate_metrics(y_true, y_pred):
    """Calculate various evaluation metrics."""
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {
        'mape': mape,
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }

def print_evaluation_results(metrics, comparison_df):
    """Print detailed evaluation results."""
    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
    print(f"R-squared Score: {metrics['r2']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.2f} rupees")
    print(f"Root Mean Squared Error: {metrics['rmse']:.2f} rupees")
    
    print("\nPrediction Analysis:")
    print("\nSummary Statistics:")
    print(comparison_df.describe())
    
    print("\nPercentile Analysis:")
    percentiles = [10, 25, 50, 75, 90]
    print(comparison_df['Percentage Error'].quantile(np.array(percentiles)/100))