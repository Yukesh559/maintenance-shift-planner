"""
Preprocessing and Feature Engineering Module
Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI ID 601)
Reference: Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Raw feature definitions
IDENTIFIER_COLS = ["UDI", "Product ID"]
CATEGORICAL_COLS = ["Type"]
NUMERICAL_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
ENGINEERED_COLS = [
    "Temp_diff [K]",
    "Power [W]",
    "Overstrain_factor [min*Nm]"
]
TARGET_BINARY = "Machine failure"
TARGET_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-derived physical features based on milling machine physics:
    1. Temp_diff: Process temperature minus Air temperature.
       (Heat Dissipation Failure occurs when Temp_diff < 8.6 K and speed < 1380 rpm)
    2. Power: Mechanical power = Torque (Nm) * Rotational speed (rad/s).
       (Power Failure occurs when Power < 3500 W or > 9000 W)
    3. Overstrain_factor: Tool wear (min) * Torque (Nm).
       (Overstrain Failure occurs when wear * torque exceeds threshold 11000 - 13000 min*Nm)
    """
    df = df.copy()
    
    # 1. Temperature difference
    df["Temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    
    # 2. Mechanical Power (W) = Torque [Nm] * (Rotational speed [rpm] * 2 * pi / 60)
    df["Power [W]"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * (2 * np.pi / 60))
    
    # 3. Overstrain factor
    df["Overstrain_factor [min*Nm]"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    
    return df


def get_feature_transformer() -> ColumnTransformer:
    """
    Creates the scikit-learn ColumnTransformer for preprocessing.
    Categorical: OneHotEncoder (Type L, M, H)
    Numerical: StandardScaler
    """
    all_num_cols = NUMERICAL_COLS + ENGINEERED_COLS
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), all_num_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def clean_and_split_data(
    data_path: str = "data/ai4i2020.csv",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Loads raw AI4I dataset, drops UDI & Product ID, engineers physical features,
    and performs a stratified split into Train (70%), Validation (15%), and Test (15%).
    Fits preprocessor ON TRAINING SET ONLY.
    
    Returns preprocessed splits, labels, fitted preprocessor, and feature names.
    """
    import os
    if not os.path.exists(data_path):
        from data.download_data import download_and_extract
        print(f"[INFO] Dataset not found at {data_path}. Auto-downloading...")
        download_and_extract(os.path.dirname(data_path) or "data")

    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Split sizes must sum to 1.0"
    
    df = pd.read_csv(data_path)
    
    # Strict compliance: Drop identifiers
    df_clean = df.drop(columns=[col for col in IDENTIFIER_COLS if col in df.columns])
    
    # Separate targets
    y_binary = df_clean[TARGET_BINARY].values
    y_modes = df_clean[TARGET_MODES].values
    
    # Feature columns before transform
    feature_cols = [c for c in df_clean.columns if c not in [TARGET_BINARY] + TARGET_MODES]
    X_raw = df_clean[feature_cols].copy()
    
    # Engineer physics features
    X_engineered = engineer_physics_features(X_raw)
    
    # Stratified split: First split Train+Val (85%) and Test (15%)
    X_train_val, X_test_raw, y_train_val, y_test, modes_train_val, modes_test = train_test_split(
        X_engineered, y_binary, y_modes,
        test_size=test_size,
        stratify=y_binary,
        random_state=random_state
    )
    
    # Second split: Train (70/85 = ~82.35%) and Val (15/85 = ~17.65%)
    val_relative_ratio = val_size / (train_size + val_size)
    X_train_raw, X_val_raw, y_train, y_val, modes_train, modes_val = train_test_split(
        X_train_val, y_train_val, modes_train_val,
        test_size=val_relative_ratio,
        stratify=y_train_val,
        random_state=random_state
    )
    
    # Fit preprocessor strictly on training data
    preprocessor = get_feature_transformer()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Retrieve feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS).tolist()
    num_feature_names = NUMERICAL_COLS + ENGINEERED_COLS
    all_feature_names = cat_feature_names + num_feature_names
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "modes_train": modes_train,
        "X_val": X_val,
        "y_val": y_val,
        "modes_val": modes_val,
        "X_test": X_test,
        "y_test": y_test,
        "modes_test": modes_test,
        "preprocessor": preprocessor,
        "feature_names": all_feature_names,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw
    }


def validate_input_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validates batch input format for Requirement C.
    Checks for required sensor columns and valid ranges.
    """
    required_cols = ["Type"] + NUMERICAL_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required sensor column(s): {', '.join(missing)}"
    
    # Check Type validity
    invalid_types = df[~df["Type"].isin(["L", "M", "H"])]["Type"].unique().tolist()
    if invalid_types:
        return False, f"Invalid machine Type found: {invalid_types}. Allowed values are 'L', 'M', or 'H'."
    
    # Check numeric ranges
    for col in NUMERICAL_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' contains non-numeric values."
        if df[col].isnull().any():
            return False, f"Column '{col}' contains missing (NaN) values."
            
    # Physical sanity checks
    if (df["Air temperature [K]"] < 250).any() or (df["Air temperature [K]"] > 350).any():
        return False, "Air temperature out of physical range [250 K, 350 K]."
    if (df["Process temperature [K]"] < 250).any() or (df["Process temperature [K]"] > 360).any():
        return False, "Process temperature out of physical range [250 K, 360 K]."
    if (df["Rotational speed [rpm]"] < 500).any() or (df["Rotational speed [rpm]"] > 3500).any():
        return False, "Rotational speed out of physical range [500 rpm, 3500 rpm]."
    if (df["Torque [Nm]"] < 0).any() or (df["Torque [Nm]"] > 150).any():
        return False, "Torque out of physical range [0 Nm, 150 Nm]."
    if (df["Tool wear [min]"] < 0).any() or (df["Tool wear [min]"] > 400).any():
        return False, "Tool wear out of physical range [0 min, 400 min]."
        
    return True, None


def preprocess_inference_data(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Transforms inference data (single machine or batch) using the fitted preprocessor.
    Safely ignores UDI and Product ID if present.
    """
    # Create working copy and remove identifier columns if present
    df_clean = df.copy()
    for col in IDENTIFIER_COLS:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            
    # Engineer physical features
    df_engineered = engineer_physics_features(df_clean)
    
    # Transform
    return preprocessor.transform(df_engineered)
