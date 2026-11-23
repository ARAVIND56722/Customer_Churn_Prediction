"""
Data Preprocessing and Pipeline Creation Module.
Handles type conversions, missing value imputation, one-hot encoding, feature scaling,
and stratified train-test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_COLS = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

def load_and_clean_data(csv_path="data/telco_customer_churn.csv"):
    """
    Loads raw telco customer churn data, converts types, and cleans missing values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 1. TotalCharges cleaning
    # Convert empty spaces or non-numeric entries to NaN, then fill
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    
    # Where tenure is 0, TotalCharges is legitimately 0
    zero_tenure_mask = (df['tenure'] == 0) & (df['TotalCharges'].isna())
    df.loc[zero_tenure_mask, 'TotalCharges'] = 0.0
    
    # For any remaining NaNs, fill with median
    if df['TotalCharges'].isna().any():
        median_val = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
        
    # Ensure SeniorCitizen is treated as category
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    # 2. Encode target 'Churn' to binary integer (1 for Yes, 0 for No)
    if 'Churn' in df.columns:
        y = (df['Churn'] == 'Yes').astype(int)
    else:
        y = None
        
    # 3. Features matrix (drop customerID and Churn)
    feature_cols = [c for c in df.columns if c not in ['customerID', 'Churn']]
    X = df[feature_cols].copy()
    
    return X, y, df

def get_preprocessor():
    """
    Constructs a ColumnTransformer for numerical scaling and categorical encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    return preprocessor

def get_feature_names(preprocessor, categorical_cols=CATEGORICAL_COLS, numerical_cols=NUMERICAL_COLS):
    """
    Extracts transformed feature names from the fitted preprocessor.
    """
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    return numerical_cols + cat_feature_names

def prepare_data(csv_path="data/telco_customer_churn.csv", test_size=0.20, random_state=42):
    """
    Loads data and produces stratified train and test splits.
    """
    X, y, raw_df = load_and_clean_data(csv_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, raw_df

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, df = prepare_data()
    print(f"Dataset successfully prepared.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Train Churn Rate: {y_train.mean():.4f}, Test Churn Rate: {y_test.mean():.4f}")

