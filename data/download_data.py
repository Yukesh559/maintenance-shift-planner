"""
Dataset Acquisition & Verification Script
Dataset: AI4I 2020 Predictive Maintenance Dataset
Source: UCI Machine Learning Repository (ID 601)
DOI/Reference: Stephan Matzka, 'Explainable Artificial Intelligence for Predictive Maintenance
Applications with a Digital Twin', IEEE ICAIGE, 2020.
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
Note: This is a synthetic dataset representing realistic milling machine conditions.
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
EXPECTED_ROWS = 10000
EXPECTED_COLS = 14

def download_and_extract(data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "ai4i2020.csv")

    if os.path.exists(csv_path):
        print(f"[INFO] Dataset already present at: {csv_path}")
        df = pd.read_csv(csv_path)
        if len(df) == EXPECTED_ROWS:
            print(f"[INFO] Verified existing dataset: {len(df)} rows, {df.shape[1]} columns.")
            _create_sample_batch(df, data_dir)
            return csv_path
        else:
            print(f"[WARN] Existing dataset corrupted or incomplete ({len(df)} rows). Re-downloading...")

    print(f"[INFO] Downloading AI4I 2020 dataset from {DATA_URL}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(DATA_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Find csv in archive
        csv_candidates = [n for n in z.namelist() if n.endswith(".csv")]
        if not csv_candidates:
            raise FileNotFoundError("No CSV file found in downloaded archive.")
        csv_filename = csv_candidates[0]
        z.extract(csv_filename, data_dir)
        extracted_path = os.path.join(data_dir, csv_filename)
        if extracted_path != csv_path:
            if os.path.exists(csv_path):
                os.remove(csv_path)
            os.rename(extracted_path, csv_path)

    df = pd.read_csv(csv_path)
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    print(f"[INFO] Successfully downloaded and verified dataset ({len(df)} rows, {df.shape[1]} columns).")

    _create_sample_batch(df, data_dir)
    return csv_path

def _create_sample_batch(df: pd.DataFrame, data_dir: str) -> str:
    """
    Creates a realistic 50-machine shift batch for testing Requirement C.
    Includes both operational machines and machines with varied failure indicators.
    """
    sample_path = os.path.join(data_dir, "sample_batch.csv")
    if not os.path.exists(sample_path):
        # Mix 15 failure rows with 35 nominal rows for realistic shift planning
        failures = df[df["Machine failure"] == 1]
        nominals = df[df["Machine failure"] == 0]
        
        sample_fail = failures.sample(n=min(15, len(failures)), random_state=42)
        sample_nom = nominals.sample(n=35, random_state=42)
        sample = pd.concat([sample_fail, sample_nom]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Save without labels to simulate realistic uninspected shift sensor readings
        feature_cols = [
            "UDI", "Product ID", "Type", 
            "Air temperature [K]", "Process temperature [K]", 
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
        ]
        sample[feature_cols].to_csv(sample_path, index=False)
        print(f"[INFO] Created shift planning sample batch at: {sample_path}")
    return sample_path

if __name__ == "__main__":
    download_and_extract()
