"""
app.py
------
Gradio interface for the music recommender.
Run locally:   python app.py
Deploy:        push to HuggingFace Spaces (Gradio SDK)
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, "src")

from features import load_data, build_feature_matrix, AUDIO_FEATURES
from model import MusicRecommender

# ------------------------------------------------------------------
# Load / train model at startup
# ------------------------------------------------------------------

ARTIFACT_PATH = "artifacts/recommender.pkl"
DATA_PATH = "data/dataset.csv"


def get_recommender() -> MusicRecommender:
    if os.path.exists(ARTIFACT_PATH):
        print("Loading pre-trained recommender...")
        return MusicRecommender.load(ARTIFACT_PATH)
    print("No artifacts found — training from scratch...")
    df = load_data(DATA_PATH)
    X, scaler = build_feature_matrix(df)
    rec = MusicRecommender(n_neighbors=50)
    rec.fit(df, X, scaler)
    rec.save(ARTIFACT_PATH)
    return rec


rec = get_recommender()
ALL_TRACK_NAMES = sorted(rec.df["track_name"].unique().tolist())


# ------------------------------------------------------------------
# Radar chart helper
# ------------------------------------------------------------------

RADAR_FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness", "speechiness", "liveness",
]

def make_radar(track_name: str):
    idxs = rec._find_track(track_name)
    if not idxs:
        return go.Figure()
    row = rec.df.iloc[idxs[0]]
    values = [row[f] for f in RADAR_FEATURES] + [row[RADAR_FEATURES[0]]]  # close loop
    labels = RADAR_FEATURES + [RADAR_FEATURES[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            line_color="#1DB954",
            fillcolor="rgba(29,185,84,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#111",
            radialaxis=dict(visible=True, range=[0, 1], color="#555"),
            angularaxis=dict(color="#aaa"),
        ),
        paper_bgcolor="#1a1a1a",
        font_color="#eee",
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
    )
    return fig


# ------------------------------------------------------------------
# Recommendation logic wired to Gradio
# ------------------------------------------------------------------

def recommend_single(track_name: str, top_n: int, same_genre: bool):
    if not track_name:
        return pd.DataFrame(), go.Figure()
    try:
        recs = rec.recommend_from_track(track_name, top_n=top_n, same_genre_only=same_genre)
        fig = make_radar(track_name)
        return recs[["track_name", "artists", "track_genre", "similarity", "popularity"]], fig
    except ValueError as e:
        return pd.DataFrame({"error": [str(e)]}), go.Figure()


def recommend_history(history_text: str, top_n: int):
    tracks = [t.strip() for t in history_text.split("\n") if t.strip()]
    if not tracks:
        return pd.DataFrame()
    try:
        recs = rec.recommend_from_history(tracks, top_n=top_n)
        return recs[["track_name", "artists", "track_genre", "similarity", "popularity"]]
    except ValueError as e:
        return pd.DataFrame({"error": [str(e)]})


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

with gr.Blocks(
    title="🎵 Music Recommender",
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
    ),
    css="""
    body { background: #111; }
    .gradio-container { max-width: 900px; margin: auto; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🎵 Music Recommender
        Find tracks that *sound* like the ones you love — powered by Spotify audio features and KNN similarity.
        """
    )

    with gr.Tab("Single Track"):
        with gr.Row():
            with gr.Column(scale=2):
                track_input = gr.Dropdown(
                    choices=ALL_TRACK_NAMES,
                    label="Seed Track",
                    info="Start typing to search",
                    filterable=True,
                )
                top_n_slider = gr.Slider(5, 20, value=10, step=1, label="Number of recommendations")
                same_genre_cb = gr.Checkbox(label="Stay within the same genre", value=False)
                run_btn = gr.Button("Find Similar Tracks", variant="primary")
            with gr.Column(scale=3):
                radar_plot = gr.Plot(label="Audio Fingerprint")

        results_table = gr.Dataframe(
            headers=["Track", "Artists", "Genre", "Similarity", "Popularity"],
            label="Recommendations",
        )
        run_btn.click(
            recommend_single,
            inputs=[track_input, top_n_slider, same_genre_cb],
            outputs=[results_table, radar_plot],
        )

    with gr.Tab("From Listening History"):
        gr.Markdown(
            "Enter one track per line. The model builds a *taste profile* "
            "from the average of their audio features and finds the nearest neighbors."
        )
        history_input = gr.Textbox(
            lines=6,
            placeholder="Can't Help Falling In Love\nI'm Yours\nSay Something",
            label="Your Listening History",
        )
        top_n_history = gr.Slider(5, 20, value=10, step=1, label="Number of recommendations")
        history_btn = gr.Button("Recommend", variant="primary")
        history_results = gr.Dataframe(
            headers=["Track", "Artists", "Genre", "Similarity", "Popularity"],
            label="Recommendations",
        )
        history_btn.click(
            recommend_history,
            inputs=[history_input, top_n_history],
            outputs=[history_results],
        )

    gr.Markdown(
        """
        ---
        **Dataset:** Spotify Tracks Dataset (114k tracks, 114 genres)  
        **Model:** K-Nearest Neighbors on normalized Spotify audio features (cosine similarity)
        """
    )

if __name__ == "__main__":
    demo.launch()
