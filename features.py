"""
features.py
-----------
Loads the dataset, cleans it, and produces the normalized
feature matrix used by the recommender.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Audio features used for similarity
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop rows with any nulls in key columns
    df = df.dropna(subset=["track_name", "artists", "track_genre"])

    # Remove exact duplicates (same track_id)
    df = df.drop_duplicates(subset="track_id").reset_index(drop=True)

    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Normalizes audio features to [0, 1] and returns the
    feature matrix X plus the fitted scaler.
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[AUDIO_FEATURES].values)
    return X, scaler


def save_artifacts(df, X, scaler, out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/tracks.csv", index=False)
    np.save(f"{out_dir}/feature_matrix.npy", X)
    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved artifacts to '{out_dir}/'")


def load_artifacts(out_dir: str = "artifacts"):
    df = pd.read_csv(f"{out_dir}/tracks.csv")
    X = np.load(f"{out_dir}/feature_matrix.npy")
    with open(f"{out_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return df, X, scaler


if __name__ == "__main__":
    df = load_data("data/dataset.csv")
    print(f"Loaded {len(df)} tracks across {df['track_genre'].nunique()} genres")
    X, scaler = build_feature_matrix(df)
    print(f"Feature matrix shape: {X.shape}")
    save_artifacts(df, X, scaler)
