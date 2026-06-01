# 🎵 Music Recommender

A content-based music recommender built on Spotify audio features.  
Given a track (or a listening history), it finds the most sonically similar songs using KNN on normalized audio features.

## How it works

1. **Feature engineering** — 9 Spotify audio features (danceability, energy, loudness, tempo, valence, acousticness, instrumentalness, speechiness, liveness) are normalized to [0, 1].
2. **KNN model** — `sklearn.NearestNeighbors` with cosine similarity finds the closest tracks in feature space.
3. **History mode** — averages the feature vectors of liked tracks into a single taste-profile centroid, then queries the KNN index.

## Dataset

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — 114k tracks across 114 genres.

Place it at `data/dataset.csv`.

## Quickstart

```bash
pip install -r requirements.txt

# Train the model (creates artifacts/)
python src/model.py

# Launch the Gradio app locally
python app.py
```

## Project structure

```
music-recommender/
├── data/
│   └── dataset.csv          # Spotify tracks dataset
├── src/
│   ├── features.py          # Data loading & feature engineering
│   └── model.py             # KNN recommender + evaluation
├── artifacts/               # Saved model & scaler (auto-created)
├── app.py                   # Gradio UI
├── requirements.txt
└── README.md
```

## Deploying to HuggingFace Spaces

1. Create a new Space (Gradio SDK).
2. Push this repo.
3. Upload `data/dataset.csv` via the Files tab or Git LFS.
4. The Space will train the model on first launch and cache it in `artifacts/`.

## Evaluation

Genre consistency is used as a proxy metric (no ground-truth user interactions in this dataset):

```python
from src.model import MusicRecommender
rec = MusicRecommender.load()
rec.evaluate_genre_consistency(sample_n=500, top_n=10)
# e.g. Genre consistency @ 10: 0.712
```

## Next steps

- [ ] Add collaborative filtering layer using user playlists (MPD dataset)
- [ ] Neural embedding via shallow autoencoder for denser latent space
- [ ] Genre-diversity toggle (penalize same-genre results)
- [ ] Artist diversity filter (avoid recommending 5 songs by the same artist)
