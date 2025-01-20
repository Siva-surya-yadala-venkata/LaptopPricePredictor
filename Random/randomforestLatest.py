import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Custom Decision Tree Implementation
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth == self.max_depth:
            return np.mean(y)

        best_feature, best_threshold, best_score, splits = None, None, float('inf'), None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_y, right_y = y[left_mask], y[right_mask]
                score = self._variance_reduction(left_y, right_y)
                if score < best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score
                    splits = (left_mask, right_mask)

        if best_score == float('inf'):
            return np.mean(y)

        left_subtree = self._build_tree(X[splits[0]], y[splits[0]], depth + 1)
        right_subtree = self._build_tree(X[splits[1]], y[splits[1]], depth + 1)
        return {"feature": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}

    def _variance_reduction(self, left_y, right_y):
        total_variance = np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)
        return total_variance

    def _traverse_tree(self, x, tree):
        if isinstance(tree, dict):
            if x[tree["feature"]] <= tree["threshold"]:
                return self._traverse_tree(x, tree["left"])
            else:
                return self._traverse_tree(x, tree["right"])
        return tree

# Custom Random Forest Implementation
class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = [] 

    def fit(self, X, y):
        for _ in range(self.n_trees):
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            bootstrap_X, bootstrap_y = X[bootstrap_indices], y[bootstrap_indices]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

# Load the dataset
data = pd.read_csv('laptop_data.csv')

# Preprocessing
data.dropna(inplace=True)
data['Inches'] = pd.to_numeric(data['Inches'], errors='coerce')
data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

# Remove outliers in Price (1st and 99th percentiles)
data = data[(data['Price'] > data['Price'].quantile(0.01)) & 
            (data['Price'] < data['Price'].quantile(0.99))]

# Normalize numerical features
scaler = StandardScaler()
data[['Inches', 'Weight']] = scaler.fit_transform(data[['Inches', 'Weight']])

# Handle categorical features with one-hot encoding
data = pd.get_dummies(data, columns=['Company', 'TypeName', 'ScreenResolution', 'Cpu',
                                     'Ram', 'Memory', 'Gpu', 'OpSys'], drop_first=True)

# Split features and target
X = data.drop(['Price'], axis=1).values
y = data['Price'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Custom Random Forest Model
rf = RandomForest(n_trees=10, max_depth=10, min_samples_split=5)
rf.fit(X_train, y_train)

# Predict and Evaluate on the test set
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Predict prices for all rows in the dataset (not just the test set)
y_all_pred = rf.predict(X)

# Add predicted prices to the original dataset
data['Predicted_Price'] = y_all_pred

# Create a DataFrame with Actual and Predicted Prices for all rows
results_df_all = data[['Price', 'Predicted_Price']]

# Rename columns for clarity
results_df_all.columns = ['Actual_Price', 'Predicted_Price']

# Save the DataFrame to a new CSV file
results_df_all.to_csv('actual_vs_predicted_prices_all.csv', index=False)

print("CSV file 'actual_vs_predicted_prices_all.csv' has been created with actual and predicted prices for all rows.")

# Scatter plot between observed and predicted values
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title("Scatter Plot: Observed vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

# Predict Price from User Input
def predict_price_from_user():
    print("\nEnter details for price prediction:")
    try:
        inches = float(input("Screen size in inches (e.g., 15.6): "))
        weight = float(input("Weight in kg (e.g., 1.5): "))
        company = input("Company (e.g., Apple): ")
        typename = input("Type (e.g., Ultrabook): ")
        screen_res = input("Screen Resolution (e.g., 1920x1080): ")
        cpu = input("CPU (e.g., Intel Core i5): ")
        ram = input("RAM (e.g., 8GB): ")
        memory = input("Memory (e.g., 256GB SSD): ")
        gpu = input("GPU (e.g., Intel Iris Plus Graphics 640): ")
        opsys = input("Operating System (e.g., macOS): ")

        # Normalize numerical features
        input_data = {
            'Inches': (inches - scaler.mean_[0]) / scaler.scale_[0],
            'Weight': (weight - scaler.mean_[1]) / scaler.scale_[1],
            **{f'Company_{company}': 1},
            **{f'TypeName_{typename}': 1},
            **{f'ScreenResolution_{screen_res}': 1},
            **{f'Cpu_{cpu}': 1},
            **{f'Ram_{ram}': 1},
            **{f'Memory_{memory}': 1},
            **{f'Gpu_{gpu}': 1},
            **{f'OpSys_{opsys}': 1},
        }

        # Ensure missing columns are set to 0
        for col in data.columns[:-1]:
            if col not in input_data:
                input_data[col] = 0

        input_vector = pd.DataFrame([input_data]).reindex(columns=data.columns[:-1], fill_value=0).values
        predicted_price = rf.predict(input_vector)[0]

        print(f"The predicted price of the laptop is: ₹{predicted_price:.2f}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Ensure the function is called outside any conditions
predict_price_from_user()
