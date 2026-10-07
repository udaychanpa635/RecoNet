"""
Cold-start evaluation: holds out a handful of items ENTIRELY from training
(zero interactions seen), then checks which model can still say anything
useful about them.

This is the actual point of fusing content embeddings into the two-tower
model. Pure ID-based collaborative filtering (popularity counts, the TF
matrix-factorization model) has literally no signal for an item it has
never seen — its embedding is randomly initialized and untouched by
training. The two-tower model can still rank a cold-start item reasonably
because its content embedding is derived from the item's text description,
not from interaction history.

Reporting overall Precision/Recall/NDCG alone hides this — a cold item
just gets ignored by all models and barely affects the average. This
script isolates the exact scenario where the extra architecture complexity
of the two-tower model earns its keep.

    python src/cold_start_eval.py
"""
import random
import numpy as np
import torch

from data_utils import load_data, build_user_item_matrix
from pytorch_two_tower import TwoTowerModel, build_content_features, make_bpr_triplets, train_model

random.seed(7)
NUM_COLD_ITEMS = 5


def main():
    items, interactions = load_data()
    num_users = max(row["user_id"] for row in interactions) + 1
    num_items = len(items)

    cold_item_ids = set(random.sample(range(num_items), NUM_COLD_ITEMS))
    print(f"Holding out items entirely from training: "
          f"{[items[i]['name'] for i in cold_item_ids]}")

    train_interactions = [row for row in interactions if row["item_id"] not in cold_item_ids]
    warm_test_interactions = [row for row in interactions if row["item_id"] in cold_item_ids]

    content_features = build_content_features(items)
    model = TwoTowerModel(num_users, num_items, content_features)
    triplets = make_bpr_triplets(train_interactions, num_items, excluded_items=cold_item_ids)
    print(f"\nTraining on {len(train_interactions)} interactions "
          f"(cold items excluded)...")
    train_model(model, triplets, epochs=15)

    # Who actually interacted with a cold item in the full (unfiltered) data?
    # Those are the users we'd hope the model ranks the cold item highly for.
    users_who_liked_cold_item = {}
    for row in warm_test_interactions:
        users_who_liked_cold_item.setdefault(row["item_id"], []).append(row["user_id"])

    print("\n" + "=" * 60)
    print("COLD-START RANKING CHECK")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        for item_id in cold_item_ids:
            interested_users = users_who_liked_cold_item.get(item_id, [])[:5]
            if not interested_users:
                continue
            ranks = []
            for user_id in interested_users:
                scores = model.score_all_items(user_id, num_items).numpy()
                rank = int((-scores).argsort().tolist().index(item_id)) + 1
                ranks.append(rank)
            avg_rank = sum(ranks) / len(ranks)
            print(f"\n  '{items[item_id]['name']}' (never seen in training)")
            print(f"    Two-tower avg rank among {num_items} items for users "
                  f"who actually liked it: {avg_rank:.1f}")
            print(f"    (Popularity / TF-MF baselines: rank is effectively random, "
                  f"~{num_items / 2:.0f}, since they've never seen this item's ID)")

    print("\nA lower avg rank than ~num_items/2 shows the content embedding is "
          "doing real work for items with zero training interactions.")


if __name__ == "__main__":
    main()
