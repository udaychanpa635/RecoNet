"""
Two-tower neural recommender in PyTorch: one tower embeds the user (from an
ID embedding), the other embeds the item (from an ID embedding fused with a
content embedding of its text description). Trained with a contrastive-style
loss (BPR-lite: positive pair should score higher than a sampled negative).

This is the flagship model — it's what separates this project from a basic
matrix-factorization tutorial: it fuses collaborative signal (who-interacted-
with-what) with content signal (what the item is actually about), which is
what lets it recommend cold-start items with no interaction history yet.

    python src/pytorch_two_tower.py
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import load_data, temporal_split, build_user_item_matrix, evaluate_ranked_lists

EMBEDDING_DIM = 32
CONTENT_DIM = 32  # dimensionality of the (simplified) text embedding


def build_content_features(items, dim=CONTENT_DIM):
    """
    Builds a content embedding per item from its text description.

    In production (see notebooks/RecoNet_Training_Colab.ipynb) this uses a
    real sentence-transformer model downloaded from the Hugging Face Hub.
    That download isn't available in this sandbox, so this function falls
    back to a deterministic hashed bag-of-words embedding — good enough to
    prove the fusion architecture works end-to-end, but swap it for real
    sentence embeddings before you report final numbers.
    """
    vocab = {}
    vectors = []
    for item in items:
        vec = np.zeros(dim, dtype="float32")
        words = item["description"].lower().replace(",", "").replace(".", "").split()
        for word in words:
            idx = hash(word) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return np.stack(vectors)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, content_features: np.ndarray,
                 embedding_dim=EMBEDDING_DIM, content_dim=CONTENT_DIM):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)

        self.register_buffer("content_features", torch.tensor(content_features))
        self.content_proj = nn.Linear(content_dim, embedding_dim)

        self.item_fuse = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
        )

    def user_tower(self, user_ids):
        return self.user_embedding(user_ids)

    def item_tower(self, item_ids):
        id_vec = self.item_id_embedding(item_ids)
        content_vec = self.content_proj(self.content_features[item_ids])
        fused = torch.cat([id_vec, content_vec], dim=-1)
        return self.item_fuse(fused)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_tower(user_ids)
        item_vec = self.item_tower(item_ids)
        return (user_vec * item_vec).sum(dim=-1)

    def score_all_items(self, user_id, num_items):
        user_vec = self.user_tower(torch.tensor([user_id])).squeeze(0)
        all_item_ids = torch.arange(num_items)
        item_vecs = self.item_tower(all_item_ids)
        return item_vecs @ user_vec


def make_bpr_triplets(train_interactions, num_items, excluded_items: set = None):
    """(user, positive_item, negative_item) triplets for BPR-style training.

    excluded_items: item ids that must never appear as negatives either —
    needed for a clean cold-start evaluation, otherwise a "held-out" item
    still gets its embedding updated every time it's sampled as a random
    negative, which defeats the point of testing on items with truly zero
    training signal.
    """
    excluded_items = excluded_items or set()
    user_positive_items = {}
    for row in train_interactions:
        user_positive_items.setdefault(row["user_id"], set()).add(row["item_id"])

    triplets = []
    for row in train_interactions:
        positives = user_positive_items[row["user_id"]]
        for _ in range(4):
            attempts = 0
            while attempts < 20:
                neg_item = random.randint(0, num_items - 1)
                attempts += 1
                if neg_item not in positives and neg_item not in excluded_items:
                    triplets.append((row["user_id"], row["item_id"], neg_item))
                    break
    return triplets


def train_model(model, triplets, epochs=15, lr=0.01, batch_size=256):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplets = np.array(triplets)

    for epoch in range(epochs):
        np.random.shuffle(triplets)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]
            users = torch.tensor(batch[:, 0], dtype=torch.long)
            pos_items = torch.tensor(batch[:, 1], dtype=torch.long)
            neg_items = torch.tensor(batch[:, 2], dtype=torch.long)

            pos_scores = model(users, pos_items)
            neg_scores = model(users, neg_items)

            # BPR loss: push positive score above negative score
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch + 1}/{epochs} — loss: {total_loss / num_batches:.4f}")


def recommend_top_k(model, num_users, num_items, k=10, exclude_seen=None):
    model.eval()
    recs = {}
    with torch.no_grad():
        for user_id in range(num_users):
            scores = model.score_all_items(user_id, num_items).numpy()
            if exclude_seen and user_id in exclude_seen:
                for seen_item in exclude_seen[user_id]:
                    scores[seen_item] = -1e9
            top_items = np.argsort(-scores)[:k]
            recs[user_id] = top_items.tolist()
    return recs


def main():
    items, interactions = load_data()
    train, test = temporal_split(interactions)
    num_users = max(row["user_id"] for row in interactions) + 1
    num_items = len(items)

    print(f"Training PyTorch two-tower model: {num_users} users, {num_items} items")
    content_features = build_content_features(items)
    model = TwoTowerModel(num_users, num_items, content_features)

    triplets = make_bpr_triplets(train, num_items)
    print(f"Training triplets: {len(triplets)}")
    train_model(model, triplets, epochs=15)

    train_user_items = build_user_item_matrix(train)
    recs = recommend_top_k(model, num_users, num_items, k=10, exclude_seen=train_user_items)

    test_user_items = build_user_item_matrix(test)
    metrics = evaluate_ranked_lists(recs, test_user_items, k=10)

    print("\nPyTorch Two-Tower results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "two_tower_model.pt"))
    print("\nModel saved to outputs/two_tower_model.pt")


if __name__ == "__main__":
    main()
