

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the CSV data using pandas
file_path = "laptop_data.csv"  
df = pd.read_csv(file_path)

data = df.to_dict(orient="records")

# 2. Remove the first column dynamically (based on position, not name)
for row in data:
    first_key = list(row.keys())[0]  
    del row[first_key]  

# 3. Data Cleaning: Remove outliers in 'Price'
prices = [float(row['Price']) for row in data]
sorted_prices = sorted(prices)
n = len(sorted_prices)

# Calculate percentiles manually
price_lower_limit = sorted_prices[int(0.01 * n)]
price_upper_limit = sorted_prices[int(0.99 * n)]

# Filter out rows where 'Price' is outside the limits
cleaned_data = [row for row in data if price_lower_limit <= float(row['Price']) <= price_upper_limit]

# Separate features (X) and target (y)
X = [{key: row[key] for key in row if key != 'Price'} for row in cleaned_data]
y = [float(row['Price']) for row in cleaned_data]

# 4. Manually One-Hot Encode categorical columns
def one_hot_encode(data, columns):
    encoded_data = []
    unique_values_list = {}
    for col in columns:
        unique_values = list(set([r[col] for r in data]))
        unique_values_list[col] = unique_values
    
    for row in data:
        encoded_row = []
        for col in columns:
            encoded_values = [1 if row[col] == value else 0 for value in unique_values_list[col]]
            encoded_row.extend(encoded_values)
        encoded_data.append(encoded_row)
    
    return encoded_data, unique_values_list

# Identify categorical columns (for simplicity, we assume all columns are categorical here)
categorical_cols = [key for key in X[0].keys()]
X_encoded, unique_values = one_hot_encode(X, categorical_cols)

# 5. Split data into 70% training and 30% testing
train_size = int(0.7 * len(X_encoded))
X_train, X_test = X_encoded[:train_size], X_encoded[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. Simple k-NN implementation (no external library)
def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_row in X_test:
        distances = []
        for i, train_row in enumerate(X_train):
            distance = sum((test_row[j] - train_row[j]) ** 2 for j in range(len(test_row))) ** 0.5
            distances.append((distance, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:k]
        
        avg_price = sum(neighbor[1] for neighbor in nearest_neighbors) / k
        predictions.append(avg_price)
    
    return predictions

# 7. Cross-Validation for k-NN (Manual implementation)
def cross_validation_knn(X, y, k_range, num_folds=5):
    fold_size = len(X) // num_folds
    errors = {k: [] for k in k_range}
    
    for k in k_range:
        print(f"Testing k = {k}")
        
        for fold in range(num_folds):
            validation_start = fold * fold_size
            validation_end = validation_start + fold_size
            X_train_fold = X[:validation_start] + X[validation_end:]
            y_train_fold = y[:validation_start] + y[validation_end:]
            X_val_fold = X[validation_start:validation_end]
            y_val_fold = y[validation_start:validation_end]
            
            y_pred = knn_predict(X_train_fold, y_train_fold, X_val_fold, k)
            mse = mean_squared_error(y_val_fold, y_pred)
            errors[k].append(mse)
    
    mean_errors = {k: np.mean(errors[k]) for k in k_range}
    return mean_errors

# 8. Mean Squared Error (MSE) Calculation
def mean_squared_error(y_true, y_pred):
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)

# 9. Perform Cross-Validation for best k
k_range = range(1, 10)
mean_errors = cross_validation_knn(X_train, y_train, k_range)

# 10. Plot the results to find optimal k
plt.figure(figsize=(10, 6))
plt.plot(list(mean_errors.keys()), list(mean_errors.values()), marker='o', linestyle='-', color='b')
plt.title("K-NN Cross-Validation: Optimal k")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)

mean_errors_df = pd.DataFrame({
    'k': list(mean_errors.keys()),
    'Error': list(mean_errors.values())
})
print("\nCross-Validation Results:")
print(mean_errors_df)

plt.table(cellText=mean_errors_df.values,
          colLabels=mean_errors_df.columns,
          cellLoc='center',
          loc='bottom',
          bbox=[0.0, -0.5, 1.0, 0.3])  

plt.subplots_adjust(left=0.2, bottom=0.3) 
plt.show()

# 11. Find the optimal k
optimal_k = min(mean_errors, key=mean_errors.get)
print(f"Optimal k found using cross-validation: {optimal_k}")

# 12. Train final model with optimal k and evaluate on test set
y_pred = knn_predict(X_train, y_train, X_test, optimal_k)

# 13. Plot the Predicted vs Actual Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Predicted vs Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True)
plt.show()

def mean_absolute_error(y_true, y_pred):
    return sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)

def r2_score(y_true, y_pred):
    y_mean = sum(y_true) / len(y_true)
    ss_total = sum((y_true[i] - y_mean) ** 2 for i in range(len(y_true)))
    ss_residual = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    return 1 - ss_residual / ss_total

# 14. Calculate and print final evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared Score: {r2}")


# 18. Load data for overhyped laptops and predict their prices (without considering the Price column for prediction)
overhyped_file_path = "overhyped_laptops.csv"  # Path to the overhyped laptop data CSV file
overhyped_df = pd.read_csv(overhyped_file_path)

# Assuming the overhyped data contains the same columns as the original data, including 'Price'
overhyped_data = overhyped_df.to_dict(orient="records")

# Remove the first column dynamically (similar to previous processing)
for row in overhyped_data:
    first_key = list(row.keys())[0]
    del row[first_key]

# 18.1 Remove the 'Price' column from the overhyped data before prediction
X_overhyped = [{key: row[key] for key in row if key != 'Price'} for row in overhyped_data]

# Apply the one-hot encoding to overhyped data (same as before)
X_overhyped_encoded, _ = one_hot_encode(X_overhyped, categorical_cols)

# 19. Predict the prices for the overhyped laptops (excluding 'Price' for prediction)
predicted_prices_overhyped = knn_predict(X_train, y_train, X_overhyped_encoded, optimal_k)

# 20. Extract the actual prices from overhyped data (we keep the 'Price' column for comparison)
actual_prices_overhyped = [float(row['Price']) for row in overhyped_data]

# 21. Create a bar graph comparing predicted and actual prices for overhyped laptops
laptop_names = [f"Laptop {i+1}" for i in range(len(overhyped_data))]  # Assuming 5 laptops for example
x = np.arange(len(laptop_names))  # x locations for the laptops

width = 0.35  # Bar width

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, actual_prices_overhyped, width, label='Actual Price', color='blue')
plt.bar(x + width/2, predicted_prices_overhyped, width, label='Predicted Price', color='orange')

plt.xlabel('Laptop')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices for Overhyped Laptops')
plt.xticks(x, laptop_names)
plt.legend()
plt.grid(True)

plt.show()

# Print the results
for i, laptop_name in enumerate(laptop_names):
    print(f"{laptop_name}: Actual Price = {actual_prices_overhyped[i]:.2f}, Predicted Price = {predicted_prices_overhyped[i]:.2f}")




# 15. Ask the user for their categorical values selections
def ask_user_for_categorical_values(unique_values):
    user_input = {}
    print("\nPlease select values for the following categories:")

    for col, values in unique_values.items():
        print(f"\nSelect {col}:")
        for i, value in enumerate(values):
            print(f"{i + 1}. {value}")
        
        choice = int(input(f"Enter the number corresponding to {col}: ")) - 1
        if 0 <= choice < len(values):
            user_input[col] = values[choice]
        else:
            print("Invalid choice. Using the first available option.")
            user_input[col] = values[0]
    
    return user_input

# 16. Predict the price based on user input
def predict_price(user_selections, unique_values, X_train, y_train, optimal_k):
    encoded_user_input = []
    for col in user_selections:
        value_index = unique_values[col].index(user_selections[col])
        encoded_user_input.append([1 if i == value_index else 0 for i in range(len(unique_values[col]))])
    
    encoded_user_input_flat = [item for sublist in encoded_user_input for item in sublist]
    
    predicted_price = knn_predict(X_train, y_train, [encoded_user_input_flat], optimal_k)[0]
    return predicted_price

user_selections = ask_user_for_categorical_values(unique_values)

# 17. Show the user's selections and predicted price
predicted_price = predict_price(user_selections, unique_values, X_train, y_train, optimal_k)

print("\nUser selections:")
for col, value in user_selections.items():
    print(f"{col}: {value}")

print(f"\nPredicted price for the selected laptop: {predicted_price:.2f}")
