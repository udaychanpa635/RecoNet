"""
Splits interactions by TIME, not randomly. A random split leaks future
behavior into training (you'd be predicting the past using the future,
which never happens in production) — this is one of the most common
mistakes in fresher recommender projects, and calling it out explicitly
is a legitimate, resume-worthy detail.

Also implements the popularity baseline: "recommend what's trending
overall." Every recommender project should report this number — it's the
floor any real model must clear.
"""
import json
import os
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data", "generated")


def load_data():
    with open(os.path.join(DATA_DIR, "items.json")) as f:
        items = json.load(f)
    with open(os.path.join(DATA_DIR, "interactions.json")) as f:
        interactions = json.load(f)
    return items, interactions


def temporal_split(interactions, test_fraction=0.2):
    """Interactions are already time-sorted by the generator; the last
    `test_fraction` of them (by time) become the test set."""
    split_idx = int(len(interactions) * (1 - test_fraction))
    return interactions[:split_idx], interactions[split_idx:]


def build_user_item_matrix(interactions):
    """Returns {user_id: set(item_id)} of items each user has interacted with."""
    user_items = defaultdict(set)
    for row in interactions:
        user_items[row["user_id"]].add(row["item_id"])
    return user_items


def popularity_baseline(train_interactions, k=10):
    """Returns the top-k most-interacted-with items overall — same list for every user."""
    counts = Counter(row["item_id"] for row in train_interactions)
    return [item_id for item_id, _ in counts.most_common(k)]


def evaluate_ranked_lists(recommendations_by_user: dict, test_user_items: dict, k=10):
    """
    Computes Precision@K, Recall@K, NDCG@K averaged across users who have
    at least one held-out interaction.
    recommendations_by_user: {user_id: [item_id, ...]} ranked, length >= k
    test_user_items: {user_id: set(item_id)} ground truth from the held-out period
    """
    import math

    precisions, recalls, ndcgs = [], [], []

    for user_id, true_items in test_user_items.items():
        if not true_items:
            continue
        recs = recommendations_by_user.get(user_id, [])[:k]
        if not recs:
            continue

        hits = [1 if item in true_items else 0 for item in recs]
        num_hits = sum(hits)

        precisions.append(num_hits / k)
        recalls.append(num_hits / len(true_items))

        dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
        ideal_hits = [1] * min(len(true_items), k)
        idcg = sum(1 / math.log2(idx + 2) for idx in range(len(ideal_hits))) or 1.0
        ndcgs.append(dcg / idcg)

    n = max(len(precisions), 1)
    return {
        f"precision@{k}": sum(precisions) / n,
        f"recall@{k}": sum(recalls) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
        "num_users_evaluated": n,
    }


if __name__ == "__main__":
    items, interactions = load_data()
    train, test = temporal_split(interactions)
    print(f"Train interactions: {len(train)} | Test interactions: {len(test)}")

    top_k = popularity_baseline(train, k=10)
    print(f"Popularity baseline top-10 item_ids: {top_k}")

    test_user_items = build_user_item_matrix(test)
    recs_by_user = {uid: top_k for uid in test_user_items}  # same list for everyone
    metrics = evaluate_ranked_lists(recs_by_user, test_user_items, k=10)

    print("\nPopularity baseline results (the floor to beat):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
