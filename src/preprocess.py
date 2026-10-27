"""Data Preprocessing and Feature Engineering Module.

Provides data cleaning, outlier treatment, feature engineering,
feature importance scoring (Mutual Information / ANOVA),
and scikit-learn ColumnTransformer pipelines.
"""

import os
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, f_classif
import joblib


NUMERICAL_FEATURES = [
    'Temperature',
    'Humidity',
    'Wind Speed',
    'Precipitation (%)',
    'Atmospheric Pressure',
    'UV Index',
    'Visibility (km)'
]

CATEGORICAL_FEATURES = [
    'Cloud Cover',
    'Season',
    'Location'
]

TARGET_COL = 'Weather Type'


def clip_physical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip physically implausible values (e.g., humidity > 100%, precipitation > 100%).

    Ensures data consistency without dropping observations.
    """
    df = df.copy()
    if 'Humidity' in df.columns:
        df['Humidity'] = df['Humidity'].clip(lower=0, upper=100)
    if 'Precipitation (%)' in df.columns:
        df['Precipitation (%)'] = df['Precipitation (%)'].clip(lower=0, upper=100)
    if 'Temperature' in df.columns:
        df['Temperature'] = df['Temperature'].clip(lower=-40, upper=60)
    if 'Visibility (km)' in df.columns:
        df['Visibility (km)'] = df['Visibility (km)'].clip(lower=0, upper=50)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive meteorologically motivated features to enhance model discrimination.

    1. Dew Point Approximation (Magnus formula approximation)
    2. Temperature-Humidity Index (Comfort / Heat Index indicator)
    3. Pressure Anomaly (deviation from standard sea-level pressure 1013.25 hPa)
    4. Freezing Indicator (binary flag for sub-zero temperatures)
    """
    df = df.copy()
    
    # Simple Dew Point approximation: T - ((100 - RH) / 5)
    df['Dew_Point_Approx'] = df['Temperature'] - ((100.0 - df['Humidity']) / 5.0)
    
    # Temperature-Humidity Index (THI): 0.8 * T + (RH/100) * (T - 14.4) + 46.4
    df['Temp_Humidity_Index'] = (
        0.8 * df['Temperature'] + (df['Humidity'] / 100.0) * (df['Temperature'] - 14.4) + 46.4
    )
    
    # Atmospheric Pressure anomaly from standard 1013.25 hPa
    df['Pressure_Anomaly'] = df['Atmospheric Pressure'] - 1013.25
    
    # Freezing condition flag (useful for Snow classification)
    df['Is_Freezing'] = (df['Temperature'] <= 0).astype(int)
    
    return df


def get_preprocessor(include_engineered: bool = True) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build scikit-learn ColumnTransformer for numerical and categorical pipelines."""
    num_cols = list(NUMERICAL_FEATURES)
    if include_engineered:
        num_cols.extend(['Dew_Point_Approx', 'Temp_Humidity_Index', 'Pressure_Anomaly', 'Is_Freezing'])
        
    cat_cols = list(CATEGORICAL_FEATURES)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor, num_cols, cat_cols


def compute_feature_importance_ranking(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """Calculate Mutual Information and ANOVA F-statistic for features."""
    mi_scores = mutual_info_classif(X_train_proc, y_train, random_state=42)
    f_scores, p_values = f_classif(X_train_proc, y_train)
    
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Mutual_Information': mi_scores,
        'ANOVA_F_Score': f_scores,
        'p_value': p_values
    }).sort_values(by='Mutual_Information', ascending=False).reset_index(drop=True)
    
    return fi_df


def prepare_data(
    file_path: str = "data/weather_classification_data.csv",
    test_size: float = 0.20,
    random_state: int = 42,
    apply_clipping: bool = True,
    apply_engineering: bool = True
) -> Dict[str, Any]:
    """Execute complete preprocessing workflow, returning split sets, preprocessor, and encoders."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    if apply_clipping:
        df = clip_physical_outliers(df)
        
    if apply_engineering:
        df = engineer_features(df)
        
    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]
    
    # Label Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    
    # Train-Test Split (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Preprocessor
    preprocessor, num_cols, cat_cols = get_preprocessor(include_engineered=apply_engineering)
    
    # Fit preprocessor on train set only to prevent data leakage
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    
    # Feature Importance analysis
    feature_importance_df = compute_feature_importance_ranking(
        X_train_transformed, y_train, all_feature_names
    )
    
    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train_proc': X_train_transformed,
        'X_test_proc': X_test_transformed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder,
        'class_names': class_names,
        'feature_names': all_feature_names,
        'feature_importance_df': feature_importance_df,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }


if __name__ == "__main__":
    data_bundle = prepare_data()
    print("Preprocessed successfully.")
    print(f"X_train shape: {data_bundle['X_train_proc'].shape}")
    print(f"X_test shape:  {data_bundle['X_test_proc'].shape}")
    print(f"Classes:       {data_bundle['class_names']}")
    print("\nTop 10 Features by Mutual Information:")
    print(data_bundle['feature_importance_df'].head(10))
