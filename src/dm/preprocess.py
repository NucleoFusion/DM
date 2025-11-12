import click
import pandas as pd
from utils import (
    df_dropna,
    df_clean_all,
    encode_categorical,
    normalize_numeric
)

@click.command()
@click.argument("filepath")
def preprocess(filepath):
    """
    Preprocess the dataset for Apriori, K-Means, and Decision Tree.
    FILEPATH: Path to CSV
    """
    # Load dataset
    df = pd.read_csv(filepath)
    print(f"📂 Loaded {len(df)} rows and {len(df.columns)} columns.")
    print(df.head())

    # 1️⃣ Drop missing or empty rows
    df = df_dropna(df)
    print(f"✅ After dropping NA: {len(df)} rows remain.")
    print(df.head())

    # 2️⃣ Clean all columns (trim, lowercase, handle lists)
    df = df_clean_all(df)
    print("✅ Cleaned string and list-type values.")
    print(df.head())

    # 3️⃣ Encode categorical columns (for ML like KMeans & DecisionTree)
    df = encode_categorical(df)
    print("✅ Encoded categorical features.")
    print(df.head())

    # 4️⃣ Normalize numeric columns (for KMeans)
    df = normalize_numeric(df)
    print("✅ Normalized numeric columns.")
    print(df.head())


if __name__ == "__main__":
    preprocess()
