"""
Neural matrix factorization in TensorFlow/Keras: learns a user embedding and
an item embedding such that their dot product predicts interaction
likelihood. Trained on implicit feedback with negative sampling (users
didn't rate items 1-5; they either interacted or didn't, so we sample
items they DIDN'T interact with as negatives).

This is the "collaborative filtering, but learned end-to-end" baseline that
sits between the popularity baseline and the full two-tower PyTorch model.

    python src/tf_mf_model.py
"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_utils import load_data, temporal_split, build_user_item_matrix, evaluate_ranked_lists

EMBEDDING_DIM = 16
NEGATIVES_PER_POSITIVE = 4


def build_model(num_users: int, num_items: int, embedding_dim: int = EMBEDDING_DIM) -> keras.Model:
    user_input = keras.Input(shape=(1,), name="user_id")
    item_input = keras.Input(shape=(1,), name="item_id")

    user_embedding = keras.layers.Embedding(num_users, embedding_dim, name="user_embedding")(user_input)
    item_embedding = keras.layers.Embedding(num_items, embedding_dim, name="item_embedding")(item_input)

    user_vec = keras.layers.Flatten()(user_embedding)
    item_vec = keras.layers.Flatten()(item_embedding)

    dot = keras.layers.Dot(axes=1)([user_vec, item_vec])
    output = keras.layers.Activation("sigmoid")(dot)

    model = keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def make_training_pairs(train_interactions, num_items, negatives_per_positive=NEGATIVES_PER_POSITIVE):
    user_ids, item_ids, labels = [], [], []
    user_positive_items = {}
    for row in train_interactions:
        user_positive_items.setdefault(row["user_id"], set()).add(row["item_id"])

    for row in train_interactions:
        user_ids.append(row["user_id"])
        item_ids.append(row["item_id"])
        labels.append(1)

        positives = user_positive_items[row["user_id"]]
        negatives_sampled = 0
        attempts = 0
        while negatives_sampled < negatives_per_positive and attempts < 20:
            neg_item = random.randint(0, num_items - 1)
            attempts += 1
            if neg_item not in positives:
                user_ids.append(row["user_id"])
                item_ids.append(neg_item)
                labels.append(0)
                negatives_sampled += 1

    return np.array(user_ids), np.array(item_ids), np.array(labels, dtype="float32")


def recommend_top_k(model, num_users, num_items, k=10, exclude_seen=None):
    """Scores every item for every user and returns the top-k per user."""
    all_items = np.arange(num_items)
    recs = {}
    for user_id in range(num_users):
        user_arr = np.full_like(all_items, user_id)
        scores = model.predict([user_arr, all_items], verbose=0).flatten()
        if exclude_seen and user_id in exclude_seen:
            for seen_item in exclude_seen[user_id]:
                scores[seen_item] = -1
        top_items = np.argsort(-scores)[:k]
        recs[user_id] = top_items.tolist()
    return recs


def main():
    items, interactions = load_data()
    train, test = temporal_split(interactions)
    num_users = max(row["user_id"] for row in interactions) + 1
    num_items = len(items)

    print(f"Training TF matrix-factorization model: {num_users} users, {num_items} items")
    user_ids, item_ids, labels = make_training_pairs(train, num_items)
    print(f"Training pairs (with negatives): {len(labels)}")

    model = build_model(num_users, num_items)
    model.fit([user_ids, item_ids], labels, epochs=8, batch_size=256, verbose=2, validation_split=0.1)

    train_user_items = build_user_item_matrix(train)
    recs = recommend_top_k(model, num_users, num_items, k=10, exclude_seen=train_user_items)

    test_user_items = build_user_item_matrix(test)
    metrics = evaluate_ranked_lists(recs, test_user_items, k=10)

    print("\nTF Matrix Factorization results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "outputs"), exist_ok=True)
    model.save(os.path.join(os.path.dirname(__file__), "..", "outputs", "tf_mf_model.keras"))
    print("\nModel saved to outputs/tf_mf_model.keras")


if __name__ == "__main__":
    main()
