import os, pandas as pd

anime = pd.read_csv("data/anime.csv")
ratings = pd.read_csv("data/rating.csv")
ratings = ratings[ratings.rating != -1].copy()

# bucket ratings to feedback (0=dislike,1=like,2=love)
def to_feedback(r):
    if r <= 4: return 0
    elif r <= 7: return 1
    else: return 2
ratings["feedback"] = ratings["rating"].apply(to_feedback)

# make contiguous ids
uid_map = {u:i for i,u in enumerate(ratings.user_id.unique())}
iid_map = {a:j for j,a in enumerate(anime.anime_id.unique())}
ratings["uid"] = ratings.user_id.map(uid_map)
ratings["iid"] = ratings.anime_id.map(iid_map)
ratings["timestamp"] = ratings.groupby("uid").cumcount()

# save
outdir = "data/processed"
os.makedirs(outdir, exist_ok=True)

items = anime[["anime_id","name","genre","type","episodes","rating","members"]].copy()
items.to_parquet(f"{outdir}/items.parquet", index=False)

ratings[["uid","iid","feedback","timestamp"]].to_parquet(f"{outdir}/interactions.parquet", index=False)

