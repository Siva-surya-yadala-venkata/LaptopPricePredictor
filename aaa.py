import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('laptop_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Drop any rows with missing values
data.dropna(inplace=True)

# Convert relevant columns to numeric types
data['Inches'] = pd.to_numeric(data['Inches'], errors='coerce')
data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)  # Remove 'kg' and convert to float
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys'], drop_first=True)

# Define features and target variable (exclude target variable)
X = data.drop(['Price'], axis=1).values  # Features
y = data['Price'].values  # Target variable

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model's performance on the test set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Function to predict price based on user input
def predict_price(features):
    features_array = np.array(features).reshape(1, -1)  # Reshape for a single sample prediction
    predicted_price = model.predict(features_array)
    return predicted_price[0]

# Example of how to use the function with user input
if __name__ == "__main__":
    print("Enter laptop features for price prediction:")
    
    # Input features from the user
    inches = float(input("Screen size in inches (e.g., 15.6): "))
    weight = float(input("Weight in kg (e.g., 1.5): "))
    
    # Collecting categorical inputs for one-hot encoding later
    company_name = input("Enter company name (e.g., Apple): ")
    type_name = input("Enter laptop type (e.g., Ultrabook): ")
    screen_resolution = input("Enter screen resolution (e.g., 1920x1080): ")
    cpu = input("Enter CPU (e.g., Intel Core i5): ")
    ram = input("Enter RAM (e.g., 8GB): ")
    memory = input("Enter memory type (e.g., 256GB SSD): ")
    gpu = input("Enter GPU (e.g., Intel Iris Plus Graphics 640): ")
    os_name = input("Enter operating system (e.g., macOS): ")

    # Prepare one-hot encoded categorical variables based on user input
    # Create a DataFrame for user input to match training data structure
    user_input_data = {
        'Inches': inches,
        'Weight': weight,
        'Company_' + company_name: 1,
        'TypeName_' + type_name: 1,
        'ScreenResolution_' + screen_resolution: 1,
        'Cpu_' + cpu: 1,
        'Ram_' + ram: 1,
        'Memory_' + memory: 1,
        'Gpu_' + gpu: 1,
        'OpSys_' + os_name: 1,
    }

    # Create a DataFrame with zeros for all other columns
    for col in data.columns:
        if col not in user_input_data:
            user_input_data[col] = 0

    user_input_df = pd.DataFrame([user_input_data])

    # Ensure that columns match the model's expected input shape
    user_input_features = user_input_df.reindex(columns=data.columns[:-1], fill_value=0).values

    predicted_price = predict_price(user_input_features)
    
    print(f"The predicted price of the laptop is: {predicted_price:.2f}")