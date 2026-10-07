"""
RecoNet API — get ranked recommendations for a user, each with a grounded
explanation from the LangChain retrieval layer.

Run:
    uvicorn api.main:app --reload --port 8001

Endpoints:
    GET  /recommend/{user_id}   -> ranked items + explanations
    GET  /users                 -> sample user ids to try
    GET  /health
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data_utils import load_data, build_user_item_matrix
from explainer import build_item_vectorstore, explain_recommendation

app = FastAPI(title="RecoNet API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Loaded once at startup — small enough for a demo; swap for a real
# model registry / feature store for anything beyond that.
ITEMS, INTERACTIONS = load_data()
ITEMS_BY_ID = {item["item_id"]: item for item in ITEMS}
USER_ITEM_HISTORY = build_user_item_matrix(INTERACTIONS)
VECTORSTORE = build_item_vectorstore(ITEMS)

# Simple content-similarity "model" for the demo API: score every item a
# user hasn't seen by how similar it is to the average of what they've
# already liked. Swap this for the trained two_tower model's score_all_items
# once you've run training locally (see src/pytorch_two_tower.py).
import numpy as np
from explainer import HashedBOWEmbeddings

_embedder = HashedBOWEmbeddings()
_item_vectors = {item["item_id"]: np.array(_embedder.embed_query(item["description"])) for item in ITEMS}


def score_items_for_user(user_id: int):
    liked_ids = USER_ITEM_HISTORY.get(user_id, set())
    if not liked_ids:
        # cold user: fall back to nothing personalized, caller handles this
        return {}
    liked_vecs = [_item_vectors[i] for i in liked_ids]
    user_profile = np.mean(liked_vecs, axis=0)

    scores = {}
    for item_id, vec in _item_vectors.items():
        if item_id in liked_ids:
            continue
        denom = (np.linalg.norm(user_profile) * np.linalg.norm(vec)) or 1e-9
        scores[item_id] = float(np.dot(user_profile, vec) / denom)
    return scores


@app.get("/health")
def health():
    return {"status": "ok", "num_items": len(ITEMS), "num_users": len(USER_ITEM_HISTORY)}


@app.get("/users")
def sample_users(limit: int = 10):
    return {"user_ids": list(USER_ITEM_HISTORY.keys())[:limit]}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 5):
    if user_id not in USER_ITEM_HISTORY:
        raise HTTPException(status_code=404, detail=f"Unknown user_id {user_id}. Try GET /users for valid ids.")

    scores = score_items_for_user(user_id)
    if not scores:
        top_items = [ITEMS[i] for i in range(min(k, len(ITEMS)))]
        return {
            "user_id": user_id,
            "recommendations": [
                {"item": item, "explanation": "New user — showing a general starting selection."}
                for item in top_items
            ],
        }

    top_item_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    liked_items = [ITEMS_BY_ID[i] for i in USER_ITEM_HISTORY[user_id]]

    recommendations = []
    for item_id in top_item_ids:
        item = ITEMS_BY_ID[item_id]
        explanation = explain_recommendation(VECTORSTORE, item, liked_items, use_llm=False)
        recommendations.append({
            "item": item,
            "score": round(scores[item_id], 4),
            "explanation": explanation,
        })

    return {"user_id": user_id, "recommendations": recommendations}


@app.get("/")
def root():
    return {
        "service": "RecoNet API",
        "endpoints": ["/recommend/{user_id} (GET)", "/users (GET)", "/health (GET)"],
        "docs": "/docs",
    }
