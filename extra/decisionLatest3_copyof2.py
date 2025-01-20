import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=10, min_samples_split=5, min_mse_decrease=1e-4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_mse_decrease = min_mse_decrease
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for row in X])

    def _build_tree(self, X, y, depth):
        if len(X) < self.min_samples_split or depth >= self.max_depth:
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if not best_split or best_split['mse'] < self.min_mse_decrease:
            return np.mean(y)

        left_idxs, right_idxs = best_split["left_idxs"], best_split["right_idxs"]

        left_tree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_tree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree,
        }

    def _find_best_split(self, X, y):
        best_split = None
        min_mse = float("inf")
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_idxs = np.where(X[:, feature] <= threshold)[0]
                right_idxs = np.where(X[:, feature] > threshold)[0]

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                mse = self._calculate_weighted_mse(y[left_idxs], y[right_idxs])

                if mse < min_mse:
                    min_mse = mse
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left_idxs": left_idxs,
                        "right_idxs": right_idxs,
                        "mse": mse,
                    }

        return best_split

    def _calculate_weighted_mse(self, y_left, y_right):
        n_left, n_right = len(y_left), len(y_right)
        mse_left = np.var(y_left) * n_left
        mse_right = np.var(y_right) * n_right
        return (mse_left + mse_right) / (n_left + n_right)

    def _predict_one(self, row, tree):
        if not isinstance(tree, dict):
            return tree
        feature, threshold = tree["feature"], tree["threshold"]
        if row[feature] <= threshold:
            return self._predict_one(row, tree["left"])
        else:
            return self._predict_one(row, tree["right"])


# Load and preprocess the dataset
data = pd.read_csv('laptop_data.csv')

# Keep a copy of the original categorical columns
original_company = data['Company'].astype('category')
original_type = data['TypeName'].astype('category')
original_os = data['OpSys'].astype('category')

# Feature Engineering: Add PPI (Pixels Per Inch)
data['Inches'] = pd.to_numeric(data['Inches'], errors='coerce')
data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)
data['Ram'] = data['Ram'].str.replace('GB', '').astype(int)
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
data['Touchscreen'] = data['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
data['IPS'] = data['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Convert screen resolution into total pixels and calculate PPI
data['Resolution'] = data['ScreenResolution'].str.extract(r'(\d+x\d+)')
data[['Width', 'Height']] = data['Resolution'].str.split('x', expand=True).astype(float)
data['PPI'] = ((data['Width'] ** 2 + data['Height'] ** 2) ** 0.5) / data['Inches']

# Convert categorical columns to category type and then encode them as integers
data['Company'] = original_company.cat.codes
data['TypeName'] = original_type.cat.codes
data['OpSys'] = original_os.cat.codes

# Define features and target variable
X = data[['Inches', 'Weight', 'Company', 'TypeName', 'OpSys', 'Ram', 'Touchscreen', 'IPS', 'PPI']].values
y = data['Price'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree
model = DecisionTreeRegressorScratch(max_depth=10, min_samples_split=10)
model.fit(X_train, y_train)

# Randomly select 10 samples for prediction and observation
random_idxs = np.random.choice(len(X_test), size=10, replace=False)
random_samples = X_test[random_idxs]
actual_prices = y_test[random_idxs]
predicted_prices = model.predict(random_samples)

# Calculate errors and additional metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
error_percentage = (abs(actual_prices - predicted_prices) / actual_prices) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Print actual and predicted prices
print("\nActual vs Predicted Prices:")
for i, idx in enumerate(random_idxs):
    print(f"Laptop {i+1}:")
    print(f"Actual Price: {actual_prices[i]}")
    print(f"Predicted Price: {predicted_prices[i]}")
    print(f"Error Percentage: {error_percentage[i]:.2f}%")
    print("-" * 30)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Prices', marker='o')
plt.plot(predicted_prices, label='Predicted Prices', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()
