"""
model.py
--------
Trains and wraps a KNN-based content recommender.
Supports single-track lookup and history-based (centroid) recommendations.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from features import AUDIO_FEATURES, load_data, build_feature_matrix, save_artifacts, load_artifacts
import os


class MusicRecommender:
    def __init__(self, n_neighbors: int = 20, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 because query track is its own neighbor
            metric=metric,
            algorithm="brute",
            n_jobs=-1,
        )
        self.df = None
        self.X = None
        self.scaler = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, X: np.ndarray, scaler):
        self.df = df.reset_index(drop=True)
        self.X = X
        self.scaler = scaler
        self.model.fit(X)
        print(f"Model fitted on {len(df)} tracks.")
        return self

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _find_track(self, track_name: str) -> list[int]:
        """Returns all row indices matching the track name (case-insensitive)."""
        mask = self.df["track_name"].str.lower() == track_name.lower()
        return self.df[mask].index.tolist()

    def _build_user_profile(self, track_names: list[str]) -> np.ndarray:
        """
        Averages the feature vectors of the given tracks to produce
        a single taste-profile vector.
        """
        idxs = []
        for name in track_names:
            found = self._find_track(name)
            if found:
                idxs.append(found[0])
        if not idxs:
            raise ValueError("None of the provided track names were found in the dataset.")
        return self.X[idxs].mean(axis=0, keepdims=True)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend_from_track(
        self,
        track_name: str,
        top_n: int = 10,
        same_genre_only: bool = False,
    ) -> pd.DataFrame:
        """Recommend tracks similar to a single seed track."""
        idxs = self._find_track(track_name)
        if not idxs:
            raise ValueError(f"Track '{track_name}' not found.")
        seed_idx = idxs[0]
        seed_vector = self.X[seed_idx].reshape(1, -1)
        seed_genre = self.df.at[seed_idx, "track_genre"]

        return self._query(
            seed_vector,
            top_n=top_n,
            exclude_idxs={seed_idx},
            genre_filter=seed_genre if same_genre_only else None,
        )

    def recommend_from_history(
        self,
        track_names: list[str],
        top_n: int = 10,
        same_genre_only: bool = False,
    ) -> pd.DataFrame:
        """
        Recommend tracks based on a list of previously-liked tracks.
        Computes the centroid of all seed vectors as the user profile.
        """
        exclude_idxs = set()
        for name in track_names:
            exclude_idxs.update(self._find_track(name))

        profile = self._build_user_profile(track_names)

        return self._query(
            profile,
            top_n=top_n,
            exclude_idxs=exclude_idxs,
            genre_filter=None,
        )

    def _query(
        self,
        vector: np.ndarray,
        top_n: int,
        exclude_idxs: set,
        genre_filter: str | None,
    ) -> pd.DataFrame:
        # Fetch more candidates than needed so we can filter
        k = min(self.n_neighbors + len(exclude_idxs) + 1, len(self.df))
        distances, indices = self.model.kneighbors(vector, n_neighbors=k)
        distances = distances[0]
        indices = indices[0]

        results = []
        for dist, idx in zip(distances, indices):
            if idx in exclude_idxs:
                continue
            row = self.df.iloc[idx]
            if genre_filter and row["track_genre"] != genre_filter:
                continue
            results.append(
                {
                    "track_name": row["track_name"],
                    "artists": row["artists"],
                    "track_genre": row["track_genre"],
                    "popularity": int(row["popularity"]),
                    "similarity": round(1 - dist, 4),  # cosine dist → similarity
                    "danceability": row["danceability"],
                    "energy": row["energy"],
                    "valence": row["valence"],
                    "tempo": round(row["tempo"], 1),
                }
            )
            if len(results) >= top_n:
                break

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Evaluation (genre consistency proxy metric)
    # ------------------------------------------------------------------

    def evaluate_genre_consistency(self, sample_n: int = 500, top_n: int = 10) -> float:
        """
        Samples `sample_n` tracks, gets their top-N neighbors,
        and computes the fraction that share the seed's genre.
        Higher = recommendations stay on-genre.
        """
        sample = self.df.sample(sample_n, random_state=42)
        scores = []
        for _, row in sample.iterrows():
            recs = self.recommend_from_track(row["track_name"], top_n=top_n, same_genre_only=False)
            if recs.empty:
                continue
            match = (recs["track_genre"] == row["track_genre"]).mean()
            scores.append(match)
        score = float(np.mean(scores))
        print(f"Genre consistency @ {top_n}: {score:.3f}")
        return score

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "artifacts/recommender.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved recommender to '{path}'")

    @classmethod
    def load(cls, path: str = "artifacts/recommender.pkl") -> "MusicRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------------
# Train script
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    df = load_data("data/dataset.csv")
    X, scaler = build_feature_matrix(df)
    save_artifacts(df, X, scaler)

    print("Training recommender...")
    rec = MusicRecommender(n_neighbors=50)
    rec.fit(df, X, scaler)
    rec.save()

    print("\nSample recommendation for 'Can't Help Falling In Love':")
    print(rec.recommend_from_track("Can't Help Falling In Love", top_n=5).to_string(index=False))

    print("\nEvaluating...")
    rec.evaluate_genre_consistency(sample_n=300, top_n=10)
