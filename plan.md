# Nostrade Engine: A Context-Aware Anime Recommender

## 1. Objective
Build a **sequence-based recommendation system** for anime that:
- Learns from user watch history, explicit ratings, and contextual feedback (dislike, like, love).
- Uses a **Transformer-based architecture (SASRec)** with additional context embeddings.
- Produces **top-N personalized recommendations** with explainability.
- Deploys as an interactive demo where users can log feedback in real-time.

---

## 2. Data Sources
### Provided Datasets
- **anime.csv**
  - `anime_id`: unique identifier
  - `name`: title
  - `genre`: genres (comma-separated string)
  - `type`: TV, movie, OVA, etc.
  - `episodes`: number of episodes
  - `rating`: average MAL score
  - `members`: number of members who added to list
- **rating.csv**
  - `user_id`: unique user
  - `anime_id`: anime rated
  - `rating`: explicit rating (1–10, -1 for unknown)

### Custom Feedback Dataset (to be collected)
- `user_id`: unique ID for demo users
- `anime_id`: anime presented in UI
- `feedback_type`: {dislike=1, like=2, love=3}
- `timestamp`: when feedback was given

---

## 3. Data Preparation
- **Indexing**
  - Map `user_id` → integer [0..U)
  - Map `anime_id` → integer [0..I)
- **Cleaning**
  - Drop anime with very few ratings (<10 users)
  - Bucket ratings: {1–4 = dislike, 5–7 = like, 8–10 = love} for consistency
- **Sequences**
  - Sort ratings/feedback by `timestamp` per user
  - Construct interaction sequences of max length L (e.g., 50)
- **Context Features**
  - **Feedback embedding**: 3 categories (dislike, like, love)
  - **Time embedding**: discretize into bins (month/quarter/year)
- **Content Embeddings (for cold start)**
  - Genres → multi-hot vectors
  - Synopsis (if scraped) → TF-IDF or MiniLM embeddings
  - Optionally use CLIP embeddings for anime posters

---

## 4. Model Architecture
### Base Model: SASRec (Self-Attentive Sequential Recommender)
- **Inputs**
  - Item embedding (anime_id)
  - Feedback embedding (1, 2, 3)
  - Time-bin embedding
- **Backbone**
  - Transformer encoder blocks (2–4 layers)
  - Hidden dimension: 128–256
  - Multi-head self-attention
- **Output**
  - Predict distribution over next item in sequence
- **Loss**
  - Binary cross-entropy with sampled negatives
  - Alternative: BPR loss

### Enhancements
- Pretrain item embeddings with **item2vec** from rating.csv co-occurrence
- Blend predictions with **content-based similarity** for cold-start users

---

## 5. Training
- **Batching**
  - Sliding window over sequences
  - Pad shorter sequences
- **Negative Sampling**
  - Sample k negative items per step
- **Optimization**
  - AdamW optimizer
  - Learning rate: 1e-3 with cosine decay
  - Dropout: 0.2
- **Regularization**
  - L2 weight decay
  - LayerNorm in Transformer blocks

---

## 6. Evaluation
- **Data Split**
  - For each user: train on first N-2 items, validate on next item, test on last item
- **Metrics**
  - Recall@K (K=10,20)
  - NDCG@K (K=10,20)
  - HitRate@K
- **Ablations**
  - SASRec (full) vs SASRec without feedback embeddings
  - SASRec vs baseline MF (2D matrix factorization)
  - SASRec + content hybrid vs SASRec only

---

## 7. Deployment
- **Inference**
  - Given last few interactions, rank top-N candidate anime
  - Combine SASRec scores with content similarity if cold start
- **Streamlit App**
  - UI with anime thumbnails and info
  - User scrolls and marks dislike/like/love
  - Recommendations update in real time
  - Logged events saved for incremental retraining
- **Optional Backend**
  - FastAPI microservice for batch scoring
  - Vector index (FAISS) for fast nearest-neighbor search in item embeddings

---

## 8. Explainability
- Display reasons for recommendation:
  - “Recommended because you loved Naruto and One Piece (similar users also watched Bleach).”
  - Show genre overlap or embedding similarity
- Visualize embedding space (t-SNE/PCA) for items and contexts

---

## 9. Extensions (Future Work)
- Add session-aware modeling (differentiate between binge sessions vs long breaks)
- Incorporate review text (sentiment analysis → enrich feedback signal)
- Train multimodal embeddings (combine synopsis text + poster images)
- Explore reinforcement learning with multi-armed bandits for online updates

---

## 10. Deliverables
- **Clean dataset files** (indexed, processed sequences, context mappings)
- **PyTorch model code** for SASRec with context embeddings
- **Training + evaluation scripts** with reproducible metrics
- **Interactive Streamlit demo** with feedback logging
- **Documentation**
  - Project README with explanation
  - Model diagrams
  - Results tables (baseline vs SASRec vs hybrid)
- **Portfolio polish**

---

## Understanding the 3D Approach, User Preferences, and Interaction

- **Is it 3D?** Yes in spirit. We model three axes: user history, item identity, and context (your like or love or dislike, plus time). We do not store a dense 3D cube. We learn three embedding signals and fuse them inside a Transformer that operates on sequences.
- **How will it know what the user likes?** It learns user taste from past interactions and the strength of feedback. It predicts the next item you would choose. Items that consistently follow your loved items get higher scores. We also condition on time so it can adapt to recent shifts.
- **How will the user interact with it?** A simple web app where you scroll cards, click Dislike or Like or Love, or search and seed favorites. The app updates recommendations in real time and logs events for training.

### What “3D” means here
We encode three signals at each timestep:
1. **Item embedding**: `E_item[anime_id]`
2. **Feedback embedding**: `E_fb[dislike|like|love]`
3. **Time embedding**: `E_time[month_bin or quarter]`

The model input at step t is:
```
x_t = E_item[i_t] + E_fb[f_t] + E_time[τ_t]
```
A SASRec style Transformer consumes the sequence `(x_1, x_2, ... x_T)` and outputs a user state that is used to score all candidate items. This is effectively a 3D formulation without allocating a giant user × item × context tensor.

### How it learns user preference
- **Objective**: given your recent sequence, predict the next anime you would interact with.
- **Signals**:
  - Stronger feedback like Love gets higher weight than Like. Dislike provides negative pressure through the loss.
  - Time bins let the model prefer recent themes you engage with.
- **Training data**:
  - From `rating.csv` we can bootstrap initial sequences by bucketing 1 to 4 as Dislike, 5 to 7 as Like, 8 to 10 as Love. Your live app clicks then replace this with real timestamps.
- **Loss**:
  - Sampled softmax or BPR. Positives are the actual next anime in your sequence. Negatives are random other anime. The model learns embeddings so that your positives score higher than negatives.

### How the user interacts
**Onboarding**
- Select a few known favorites by search or from a popular grid.
- Immediate first recommendations appear.

**Main loop**
- Scroll a feed of anime cards with title, genres, short synopsis, and poster.
- Actions on each card:
  - Dislike
  - Like
  - Love
  - Save for later
- Optional actions:
  - Search for a title and mark it Love to steer the model
  - Filter by genre or type
- The app logs each action:
```
user_id, anime_id, feedback_type, timestamp
```
- The recommender updates the ranked list in real time using the latest few interactions. Nightly or on demand you can fine tune the model with the new logs.

**Explainability shown on each recommendation**
- “Because you loved Attack on Titan and Code Geass”
- “High overlap in users and similar genres”
- Optionally show top similar items in the learned embedding space

### Data flow and storage
- **Online store**: recent events in a lightweight DB (SQLite or Postgres)
- **Feature maps**:
  - `anime_id` to item embedding index
  - `feedback_type` to small integer {1,2,3}
  - `timestamp` to time bin id
- **Offline training**: PyTorch model checkpoints saved regularly
- **Serving**:
  - Rank by `score(u, i) = softmax(Transformer_state_u · E_item[i])`
  - Blend with content similarity for cold start
    ```
    final = 0.8 * model_score + 0.2 * cosine(content_user, content_item)
    ```

### Minimal schemas
```
items(anime_id PK, name, genres, type, episodes, rating_mean, members)
events(user_id, anime_id, feedback_type, timestamp)
id_maps(user_id→uid_idx, anime_id→item_idx, time→time_idx)
```

### Why this stands out
- It is not a basic 2D matrix factorization. It is a sequence model with explicit context.
- You can demo short term adaptation. If the user loves a romance tonight, the feed tilts toward romance immediately.
- You can show clear ablations: remove feedback embedding or time embedding and report the drop in Recall@20 and NDCG@20.

  - Blog post / Medium article summarizing the approach
  - GitHub repo with clear commit history and instructions
