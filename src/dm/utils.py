import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def df_dropna(df):
    """Remove rows or columns with too many missing values."""
    df = df.dropna(how='all')  # drop completely empty rows
    df = df.dropna(axis=1, how='all')  # drop empty columns
    return df.reset_index(drop=True)


def df_clean_all(df):
    """Clean every cell in the dataset."""
    for col in df.columns:
        df[col] = df[col].apply(_clean_cell)
    return df


def _clean_cell(cell):
    """Clean a single cell — handle strings, lists, and commas."""
    if pd.isna(cell):
        return None

    if isinstance(cell, str):
        cell = cell.strip()
        if cell == "":
            return None

        # Try to safely evaluate list-like strings
        try:
            val = ast.literal_eval(cell)
            if isinstance(val, list):
                return [str(x).strip().lower() for x in val if str(x).strip() != ""]
            else:
                return str(val).strip().lower()
        except Exception:
            # Fallback for comma-separated strings
            if "," in cell:
                return [x.strip().lower() for x in cell.split(",") if x.strip() != ""]
            return cell.lower()

    return cell


def encode_categorical(df):
    """Encode non-numeric columns for ML algorithms."""
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' or isinstance(df[col].iloc[0], list):
            # Convert list to string for encoding
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    return df


def normalize_numeric(df):
    """Normalize all numeric columns for KMeans."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
