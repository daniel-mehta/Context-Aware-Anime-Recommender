import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# local backend
from predict import load_artifacts, recommend_topk, RecommenderArtifacts

st.set_page_config(page_title="Nostrade Engine", layout="wide")

# ---------- Cached loaders ----------

@st.cache_resource(show_spinner=True)
def get_artifacts() -> RecommenderArtifacts:
    return load_artifacts(
        items_parquet="data/processed/items.parquet",
        ckpt_path="checkpoints/sasrec_final.pt",
        iid_map_path="artifacts/iid_map.json",
        build_genres=True,
        max_len=50,
    )

art = get_artifacts()

# ---------- Header ----------

st.title("Nostrade Engine")
st.caption("Context-aware anime recommender with dislike, like, love feedback and genre-aware blending.")

# ---------- Sidebar controls ----------

with st.sidebar:
    st.subheader("Controls")
    topk = st.slider("Top K recommendations", 5, 30, 10, step=1)
    use_genre_blend = st.checkbox("Use genre blend", value=True)
    genre_alpha = st.slider("Genre blend weight α", 0.0, 0.5, 0.2, step=0.05)
    show_graph = st.checkbox("Show influence graph", value=True)

    st.divider()
    st.subheader("Filtering (for table view only)")
    search = st.text_input("Search title contains", "")
    # quick multi-select genres
    all_genres = sorted({g.strip() for row in art.items_df["genre"].dropna() for g in str(row).split(",")})
    selected_genres = st.multiselect("Filter by genre", options=all_genres, default=[])
# ---------- Build editable table ----------

items = art.items_df.copy()
cols = ["anime_id", "name", "genre", "type", "episodes", "members"]  # include members for popularity
items = items[cols]

# Rank by popularity (members)
items = items.sort_values("members", ascending=False)

# Limit to top 500 for editing table
items = items.head(500)

# Optional filter for table view
mask = pd.Series(True, index=items.index)
if search:
    mask &= items["name"].str.contains(search, case=False, na=False)
if selected_genres:
    mask &= items["genre"].fillna("").apply(lambda s: all(g in s for g in selected_genres))

items_view = items.loc[mask].reset_index(drop=True)

# Add feedback column with None default
# Map text -> numeric later
items_view["feedback"] = None

st.write("### Pick feedback per title")
st.caption("Leave blank unless you’ve watched it. Use the drop-down to set Dislike / Like / Love.")

# Map to drop-down options
feedback_options = {
    None: None,
    "Dislike": 0,
    "Like": 1,
    "Love": 2,
}

edited = st.data_editor(
    items_view,
    use_container_width=True,
    height=600,
    column_config={
        "feedback": st.column_config.SelectboxColumn(
            "Feedback",
            help="Mark only shows you know",
            options=[None, "Dislike", "Like", "Love"],  # 👈 text labels
            required=False,
            width="small",
        ),
        "episodes": st.column_config.NumberColumn(format="%d"),
    },
    disabled=["anime_id", "name", "genre", "type", "episodes", "members"],
    hide_index=True,
    num_rows="fixed",
)

# ---------- Predict button ----------
if st.button("Predict", type="primary", use_container_width=True):
    # Gather choices as (anime_id, feedback)
    choices: List[Tuple[int, int]] = []
    for _, row in edited.iterrows():
        fb = row["feedback"]
        if fb in ("Dislike", "Like", "Love"):
            fb_num = feedback_options[fb]
            choices.append((int(row["anime_id"]), fb_num))


    if len(choices) == 0:
        st.warning("Please assign at least one feedback value (0, 1, or 2) before predicting.")
        st.stop()

    with st.spinner("Scoring recommendations..."):
        res = recommend_topk(
            art=art,
            choices=choices,
            topk=topk,
            use_genre_blend=use_genre_blend,
            genre_alpha=genre_alpha,
        )

    # ---------- Show recommendations ----------
    st.success("Done")
    st.write("### Top recommendations")
    rec_df = pd.DataFrame(res["recommendations"])
    # Order columns nicely
    if not rec_df.empty:
        rec_df = rec_df[["name", "anime_id", "score"]]
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    # ---------- Simple why table ----------
    st.write("### Why these?")
    explains = res.get("explains", [])
    if explains:
        exp_df = pd.DataFrame(explains)
        # Map iids to names
        id_to_name = {i: art.items_df.iloc[i]["name"] for i in range(len(art.items_df))}
        exp_df["Recommended"] = exp_df["rec_iid"].map(id_to_name)
        exp_df["Most similar from your picks"] = exp_df["source_iid"].map(id_to_name)
        exp_df["similarity"] = exp_df["sim"]
        exp_df = exp_df[["Recommended", "Most similar from your picks", "similarity"]]
        st.dataframe(exp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No explanations to show yet. Add a few selections first.")

    # ---------- Optional influence graph ----------
    if show_graph and explains:
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            st.write("### Influence graph")
            G = nx.DiGraph()
            # Add nodes
            picked_names = set()
            rec_names = set()
            id_to_name = {i: art.items_df.iloc[i]["name"] for i in range(len(art.items_df))}
            for e in explains:
                src_name = id_to_name[e["source_iid"]]
                rec_name = id_to_name[e["rec_iid"]]
                picked_names.add(src_name)
                rec_names.add(rec_name)
                G.add_node(src_name, kind="picked")
                G.add_node(rec_name, kind="rec")
                G.add_edge(src_name, rec_name, weight=e["sim"])

            pos = nx.spring_layout(G, k=0.6, seed=42)
            plt.figure(figsize=(8, 6))
            # Draw picked nodes
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=list(picked_names),
                node_size=500,
                node_color="#7aa2f7",
                alpha=0.9,
            )
            # Draw recommended nodes
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=list(rec_names),
                node_size=500,
                node_color="#9ece6a",
                alpha=0.9,
            )
            # Edges weighted by similarity
            weights = [G[u][v]["weight"] for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=[1 + 3*w for w in weights], alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=9)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.warning(f"Could not render graph: {e}")

# ---------- Footer helper ----------

with st.expander("How to use"):
    st.markdown(
        """
1) Use the table to set feedback per title:
   - 0 = dislike
   - 1 = like
   - 2 = love  
2) Click **Predict** to get top K recommendations.  
3) The **Why these?** section shows, for each recommendation, which of your picks was most similar in the learned embedding space.  
4) Enable **Show influence graph** in the sidebar to visualize those links.
        """
    )