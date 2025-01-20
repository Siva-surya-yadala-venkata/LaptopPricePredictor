import numpy as np
import pandas as pd

def extract_numeric(series):
    """Extract numeric values from string series."""
    return pd.to_numeric(series.str.extract(r'(\d+)', expand=False), errors='coerce')

def create_comprehensive_features(data):
    """Create advanced features for laptop price prediction."""
    data_copy = data.copy()
    
    # Handle RAM
    data_copy['Ram_gb'] = data_copy['Ram'].str.extract(r'(\d+)').astype(float)
    
    # Handle Screen Inches
    data_copy['Inches'] = pd.to_numeric(data_copy['Inches'], errors='coerce')
    
    # Handle Screen Resolution
    data_copy['Resolution_Width'] = data_copy['ScreenResolution'].str.extract(r'(\d+)x').astype(float)
    data_copy['Resolution_Height'] = data_copy['ScreenResolution'].str.extract(r'x(\d+)').astype(float)
    
    # Handle Weight
    data_copy['Weight'] = data_copy['Weight'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Create interaction features
    data_copy['Inches_Ram_interaction'] = data_copy['Inches'] * data_copy['Ram_gb']
    data_copy['Resolution_Score'] = data_copy['Resolution_Width'] * data_copy['Resolution_Height'] / 1000000
    
    return data_copy

def encode_categorical(data):
    """Enhanced categorical encoding with one-hot encoding."""
    encoded_data = data.copy()
    categorical_columns = ['Company', 'TypeName', 'Cpu', 'Memory', 'Gpu', 'OpSys']
    
    for column in categorical_columns:
        one_hot = pd.get_dummies(encoded_data[column], prefix=column)
        encoded_data = pd.concat([encoded_data.drop(columns=[column]), one_hot], axis=1)
    
    return encoded_data

def normalize_numeric(data):
    """Advanced normalization with robust scaling."""
    numeric_columns = [
        'Inches', 'Ram_gb', 'Weight', 'Resolution_Score',
        'Resolution_Width', 'Resolution_Height', 'Inches_Ram_interaction'
    ]
    
    # Fill NaN with median
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    normalized_data = data[numeric_columns].copy()
    normalized_data = (normalized_data - normalized_data.median()) / (
        normalized_data.quantile(0.75) - normalized_data.quantile(0.25)
    )
    
    return normalized_data

def prepare_data(data):
    """Comprehensive data preparation pipeline."""
    # Remove unnamed column if present
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    # Validate input data
    required_columns = [
        'Company', 'TypeName', 'Inches', 'ScreenResolution',
        'Ram', 'Memory', 'Cpu', 'Gpu', 'OpSys', 'Weight', 'Price'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Create advanced features
    data_with_features = create_comprehensive_features(data)
    
    # Prepare features and target
    features = data_with_features.drop(columns=["Price", "ScreenResolution"])
    target = data_with_features["Price"].values

    # Encode and normalize
    features_encoded = encode_categorical(features)
    features_normalized = normalize_numeric(features_encoded)
    
    # Combine one-hot encoded and normalized features
    final_features = pd.concat([
        features_encoded.select_dtypes(include=['bool']), 
        features_normalized
    ], axis=1)
    
    return final_features.values, target