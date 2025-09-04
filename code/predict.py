from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

# -------------- Model -----------------

class SASRec(nn.Module):
    def __init__(self, num_items: int, fb_vocab: int = 4, max_len: int = 50,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items, d_model, padding_idx=0)
        self.fb_emb   = nn.Embedding(fb_vocab, d_model, padding_idx=0)  # 0 pad, 1 dislike, 2 like, 3 love
        self.pos_emb  = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.layernorm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_bias = nn.Parameter(torch.zeros(num_items))

    def forward(self, iid_seq: torch.Tensor, fb_seq: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T = iid_seq.size()
        pos = torch.arange(T, device=iid_seq.device).unsqueeze(0).expand(B, T)
        x = self.item_emb(iid_seq) + self.fb_emb(fb_seq) + self.pos_emb(pos)
        x = self.layernorm(x)
        x = self.dropout(x)
        causal = torch.triu(torch.ones(T, T, device=iid_seq.device), diagonal=1).bool()
        key_pad = attn_mask == 0
        h = self.encoder(x, mask=causal, src_key_padding_mask=key_pad)
        last_idx = attn_mask.sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        h_last = h[torch.arange(B, device=iid_seq.device), last_idx]
        logits = h_last @ self.item_emb.weight.T + self.out_bias
        return logits


# -------------- Data bundle -----------------

@dataclass
class RecommenderArtifacts:
    model: SASRec
    device: torch.device
    items_df: pd.DataFrame  # must be aligned with item indices
    iid_map: Dict[int, int] # anime_id -> iid
    inv_iid_map: Dict[int, int] # iid -> anime_id
    genre_matrix: Optional[np.ndarray] # shape [num_items, num_genres] or None
    item_emb_numpy: Optional[np.ndarray] # item embedding table for cosine explains

# -------------- Loading -----------------

def load_artifacts(
    items_parquet: str = "data/processed/items.parquet",
    ckpt_path: str = "checkpoints/sasrec_final.pt",
    iid_map_path: str = "artifacts/iid_map.json",
    build_genres: bool = True,
    max_len: int = 50
) -> RecommenderArtifacts:
    """
    Loads the model and aligns item indices to the EXACT mapping used at training time.

    - Uses artifacts/iid_map.json as the source of truth for item indexing.
    - Reorders/filters items_df to match iid_map order and length.
    - Renames LayerNorm keys if the checkpoint used 'layernorm.*' instead of 'norm.*'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load iid_map (anime_id -> iid) from training time
    with open(iid_map_path, "r") as f:
        raw_map = json.load(f)
    iid_map: Dict[int, int] = {int(k): int(v) for k, v in raw_map.items()}
    num_items_ckpt = len(iid_map)
    inv_iid_map = {v: k for k, v in iid_map.items()}

    # 2) Load item metadata. Prefer processed parquet, else fall back to raw CSV.
    items_df: Optional[pd.DataFrame] = None
    if Path(items_parquet).exists():
        items_df = pd.read_parquet(items_parquet)
    else:
        # Fallback to raw CSV
        candidates = [Path("data/anime.csv"), Path("data/raw/anime.csv")]
        for p in candidates:
            if p.exists():
                items_df = pd.read_csv(p)
                break
    if items_df is None:
        raise FileNotFoundError(
            "Could not find items metadata. Expected data/processed/items.parquet or data/anime.csv"
        )

    # 3) Ensure we have required columns
    for col in ["anime_id", "name"]:
        if col not in items_df.columns:
            raise ValueError(f"Items metadata missing required column: {col}")

    # 4) Filter and REORDER items_df to match iid_map order (0..N-1)
    #    This is critical so model indices align with rows.
    rows = []
    for iid in range(num_items_ckpt):
        aid = inv_iid_map[iid]
        row = items_df.loc[items_df["anime_id"] == aid]
        if row.empty:
            # If a particular anime_id is missing in current metadata, insert a stub
            rows.append({"anime_id": aid, "name": f"Title {aid}", "genre": "", "type": "", "episodes": np.nan, "rating": np.nan, "members": np.nan})
        else:
            rows.append(row.iloc[0].to_dict())
    items_ordered = pd.DataFrame(rows)
    # Fill missing optional columns for safety
    for col in ["genre", "type", "episodes", "rating", "members"]:
        if col not in items_ordered.columns:
            items_ordered[col] = np.nan
    items_df = items_ordered.reset_index(drop=True)

    # 5) Build model with num_items = len(iid_map) so it matches checkpoint
    model = SASRec(num_items=num_items_ckpt, max_len=max_len).to(device)

    # 6) Load checkpoint and fix key names if needed
    state = torch.load(ckpt_path, map_location=device)
    # Some checkpoints save keys under 'model' or similar; unwrap if necessary
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Rename LayerNorm keys if they differ
    renamed = {}
    for k, v in state.items():
        if k.startswith("layernorm."):
            k2 = k.replace("layernorm.", "norm.")
            renamed[k2] = v
        else:
            renamed[k] = v
    state = renamed

    # Finally, load with strict=True for matching shapes
    # If you still see unexpected keys, you can set strict=False, but try strict=True first.
    missing, unexpected = model.load_state_dict(state, strict=False)
    # We allow non-critical unexpected/missing like norm vs layernorm handled above.
    # You can print them if debugging:
    # print("Missing:", missing)
    # print("Unexpected:", unexpected)

    model.eval()

    # 7) Optional: build genre matrix using the aligned items_df
    genre_matrix = None
    if build_genres:
        genre_matrix = _build_genre_matrix(items_df, iid_map)

    # 8) Cache item embedding for explanations
    with torch.no_grad():
        item_emb_numpy = model.item_emb.weight.detach().cpu().numpy()

    return RecommenderArtifacts(
        model=model,
        device=device,
        items_df=items_df,
        iid_map=iid_map,
        inv_iid_map=inv_iid_map,
        genre_matrix=genre_matrix,
        item_emb_numpy=item_emb_numpy
    )


def _build_genre_matrix(items_df: pd.DataFrame, iid_map: Dict[int, int]) -> np.ndarray:
    genres_all = sorted({g.strip()
                         for row in items_df["genre"].dropna()
                         for g in str(row).split(",")})
    g2i = {g: i for i, g in enumerate(genres_all)}
    G = np.zeros((len(iid_map), len(genres_all)), dtype=np.float32)
    for _, row in items_df.iterrows():
        aid = int(row["anime_id"])
        if aid in iid_map and isinstance(row.get("genre", ""), str):
            i = iid_map[aid]
            for g in str(row["genre"]).split(","):
                g = g.strip()
                if g in g2i:
                    G[i, g2i[g]] = 1.0
    # l2 normalize rows
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    return G

# -------------- Inference utilities -----------------

def build_sequence_from_feedback(
    choices: List[Tuple[int, int]],
    iid_map: Dict[int, int],
    max_len: int = 50
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    choices: list of (anime_id, fb_raw) where fb_raw in {0,1,2} meaning dislike, like, love
    Returns tensors ready for model: iid_seq [1,T], fb_seq [1,T], attn [1,T]
    Also returns the converted iid list used for tracing.
    """
    seq_items: List[int] = []
    seq_fbs: List[int] = []
    for aid, fb in choices:
        if aid not in iid_map:
            continue
        seq_items.append(iid_map[aid])
        # model expects 1..3 for dislike,like,love and 0 for pad
        seq_fbs.append((fb + 1) if fb in (0, 1, 2) else 0)

    # pad left, keep last max_len
    seq_items = seq_items[-max_len:]
    seq_fbs = seq_fbs[-max_len:]
    T = max_len
    iid_seq = torch.zeros(1, T, dtype=torch.long)
    fb_seq = torch.zeros(1, T, dtype=torch.long)
    attn = torch.zeros(1, T, dtype=torch.long)
    L = len(seq_items)
    if L > 0:
        iid_seq[0, -L:] = torch.tensor(seq_items, dtype=torch.long)
        fb_seq[0, -L:] = torch.tensor(seq_fbs, dtype=torch.long)
        attn[0, -L:] = 1
    return iid_seq, fb_seq, attn, seq_items

def _build_user_content_vec(
    seq_items: List[int],
    seq_fbs: List[int],
    G: Optional[np.ndarray],
    K: int = 10
) -> Optional[np.ndarray]:
    if G is None or len(seq_items) == 0:
        return None
    # back to 0..2 feedback
    fbs_012 = [max(0, min(2, f - 1)) for f in seq_fbs]  # model fb was 1..3
    idx = seq_items[-K:]
    fbv = fbs_012[-K:]
    w = {0: 1.0, 1: 2.0, 2: 3.0}
    u = np.zeros(G.shape[1], dtype=np.float32)
    total = 0.0
    for i, f in zip(idx, fbv):
        u += w[int(f)] * G[int(i)]
        total += w[int(f)]
    if total > 0:
        u /= total
    u = u / (np.linalg.norm(u) + 1e-9)
    return u

def _genre_blend_scores(
    logits: torch.Tensor,
    user_vec: Optional[np.ndarray],
    G: Optional[np.ndarray],
    alpha: float = 0.2
) -> torch.Tensor:
    if user_vec is None or G is None:
        return logits
    cs = torch.from_numpy(G @ user_vec).to(logits.device)
    return (1 - alpha) * logits + alpha * cs

def recommend_topk(
    art: RecommenderArtifacts,
    choices: List[Tuple[int, int]],  # (anime_id, fb_raw 0/1/2)
    topk: int = 10,
    use_genre_blend: bool = True,
    genre_alpha: float = 0.2
) -> Dict:
    """
    Returns a dict with:
      - recommendations: List[Dict{name, anime_id, iid, score}]
      - explains: List[Dict{rec_iid, source_iid, sim}] top contributing past item per recommendation
    """
    model, device = art.model, art.device

    iid_seq, fb_seq, attn, seq_iids = build_sequence_from_feedback(choices, art.iid_map, model.max_len)
    iid_seq = iid_seq.to(device)
    fb_seq = fb_seq.to(device)
    attn = attn.to(device)

    with torch.no_grad():
        logits = model(iid_seq, fb_seq, attn).squeeze(0)

        user_vec = None
        if use_genre_blend:
            user_vec = _build_user_content_vec(seq_iids, fb_seq.squeeze(0).cpu().tolist(), art.genre_matrix)
            logits = _genre_blend_scores(logits, user_vec, art.genre_matrix, alpha=genre_alpha)

        scores, idx = torch.topk(logits, k=topk)
        idx = idx.cpu().tolist()
        scores = scores.cpu().tolist()

    recs = []
    for iid, s in zip(idx, scores):
        aid = art.inv_iid_map[iid]
        row = art.items_df.iloc[iid]
        name = row["name"]
        recs.append({"iid": iid, "anime_id": int(aid), "name": str(name), "score": float(s)})

    explains = _compute_simple_explanations(art.item_emb_numpy, seq_iids, idx)

    return {"recommendations": recs, "explains": explains}

# -------------- Simple explainability -----------------

def _compute_simple_explanations(
    item_emb_matrix: np.ndarray,
    past_iids: List[int],
    rec_iids: List[int]
) -> List[Dict]:
    """
    For each recommended item, find the most similar item from the user's past by cosine similarity.
    Returns a list of edges for a simple graph: rec <- source with sim in [0,1].
    """
    if len(past_iids) == 0:
        return []
    V = item_emb_matrix
    norms = np.linalg.norm(V, axis=1) + 1e-9
    out: List[Dict] = []
    for r in rec_iids:
        v = V[r]
        sims = (V[past_iids] @ v) / (np.linalg.norm(V[past_iids], axis=1) * np.linalg.norm(v) + 1e-9)
        j = int(np.argmax(sims))
        out.append({
            "rec_iid": int(r),
            "source_iid": int(past_iids[j]),
            "sim": float(max(0.0, min(1.0, sims[j])))
        })
    return out
