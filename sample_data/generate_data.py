"""
Generates synthetic e-commerce style interaction data: users, items (with
text descriptions), and implicit feedback (views/purchases) with a genuine
temporal signal, so a time-based train/test split is meaningful.

Real training should swap this for MovieLens-25M or Amazon Reviews (see
notebooks/RecoNet_Training_Colab.ipynb) — this exists so the whole pipeline
(baselines, TF model, PyTorch model, LangChain explainer, API) is testable
end-to-end without downloading anything.
"""
import json
import os
import random
from datetime import datetime, timedelta

random.seed(42)

CATEGORIES = {
    "electronics": ["wireless earbuds", "bluetooth speaker", "power bank", "phone case",
                     "usb-c cable", "laptop stand", "webcam", "mechanical keyboard"],
    "home": ["ceramic mug", "throw blanket", "desk lamp", "candle set", "wall clock",
             "storage bins", "kitchen scale", "cutting board"],
    "books": ["mystery novel", "sci-fi anthology", "cookbook", "self-help guide",
              "history book", "poetry collection", "biography", "travel guide"],
    "fitness": ["yoga mat", "resistance bands", "water bottle", "foam roller",
                "jump rope", "gym gloves", "protein shaker", "fitness tracker strap"],
}

DESCRIPTIONS = {
    "wireless earbuds": "Compact wireless earbuds with noise isolation and long battery life, great for commuting and workouts.",
    "bluetooth speaker": "Portable bluetooth speaker with deep bass, waterproof design, ideal for outdoor use.",
    "power bank": "High-capacity power bank for fast-charging phones and tablets on the go.",
    "phone case": "Shock-absorbent phone case with a slim profile and raised bezel for screen protection.",
    "usb-c cable": "Durable braided USB-C cable rated for fast charging and reliable data transfer.",
    "laptop stand": "Adjustable aluminum laptop stand that improves posture and airflow during long work sessions.",
    "webcam": "1080p webcam with autofocus, well suited for video calls and streaming.",
    "mechanical keyboard": "Tactile mechanical keyboard with hot-swappable switches, popular with programmers.",
    "ceramic mug": "Hand-glazed ceramic mug that keeps drinks warm longer, dishwasher safe.",
    "throw blanket": "Soft fleece throw blanket, machine washable, cozy for couch or bed.",
    "desk lamp": "LED desk lamp with adjustable brightness and color temperature for reading or work.",
    "candle set": "Set of scented soy candles with a long burn time, calming lavender and vanilla notes.",
    "wall clock": "Minimalist silent wall clock, easy to read from across the room.",
    "storage bins": "Stackable fabric storage bins for closet organization, collapsible when empty.",
    "kitchen scale": "Precise digital kitchen scale for baking and portion control.",
    "cutting board": "Bamboo cutting board, gentle on knife edges and naturally antimicrobial.",
    "mystery novel": "A tightly plotted mystery novel full of twists, perfect for a weekend read.",
    "sci-fi anthology": "A collection of short science fiction stories exploring AI, space travel, and first contact.",
    "cookbook": "A cookbook of quick weeknight recipes using everyday ingredients.",
    "self-help guide": "A practical guide to building better habits and staying focused.",
    "history book": "An accessible narrative history covering a pivotal period with vivid detail.",
    "poetry collection": "A quiet, reflective poetry collection about everyday life and memory.",
    "biography": "An in-depth biography tracing a public figure's rise, setbacks, and legacy.",
    "travel guide": "A travel guide packed with practical tips, maps, and local recommendations.",
    "yoga mat": "Non-slip yoga mat with extra cushioning, suitable for studio or home practice.",
    "resistance bands": "A set of resistance bands of varying strength for strength training anywhere.",
    "water bottle": "Insulated stainless steel water bottle that keeps drinks cold for 24 hours.",
    "foam roller": "High-density foam roller for muscle recovery and myofascial release.",
    "jump rope": "Adjustable speed jump rope for cardio and HIIT workouts.",
    "gym gloves": "Padded gym gloves that improve grip and reduce hand fatigue during lifts.",
    "protein shaker": "Leak-proof protein shaker bottle with a built-in mixing mechanism.",
    "fitness tracker strap": "Replacement silicone strap compatible with most fitness trackers.",
}

NUM_USERS = 300
NUM_DAYS = 60


def build_items():
    items = []
    item_id = 0
    for category, names in CATEGORIES.items():
        for name in names:
            items.append({
                "item_id": item_id,
                "name": name,
                "category": category,
                "description": DESCRIPTIONS[name],
            })
            item_id += 1
    return items


def build_interactions(items):
    """Each user has 1-2 preferred categories; interactions cluster around those,
    with a light trend over time so a temporal split is meaningful."""
    interactions = []
    start_date = datetime(2026, 1, 1)
    category_names = list(CATEGORIES.keys())

    for user_id in range(NUM_USERS):
        preferred = random.sample(category_names, k=random.choice([1, 2]))
        num_interactions = random.randint(8, 30)
        for _ in range(num_interactions):
            if random.random() < 0.8:
                category = random.choice(preferred)
            else:
                category = random.choice(category_names)  # occasional exploration
            candidate_items = [i for i in items if i["category"] == category]
            item = random.choice(candidate_items)
            day_offset = random.randint(0, NUM_DAYS - 1)
            timestamp = start_date + timedelta(days=day_offset, hours=random.randint(0, 23))
            interactions.append({
                "user_id": user_id,
                "item_id": item["item_id"],
                "timestamp": timestamp.isoformat(),
                "event": random.choices(["view", "purchase"], weights=[0.75, 0.25])[0],
            })

    interactions.sort(key=lambda x: x["timestamp"])
    return interactions


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(out_dir, exist_ok=True)

    items = build_items()
    interactions = build_interactions(items)

    with open(os.path.join(out_dir, "items.json"), "w") as f:
        json.dump(items, f, indent=2)
    with open(os.path.join(out_dir, "interactions.json"), "w") as f:
        json.dump(interactions, f, indent=2)

    print(f"Generated {len(items)} items and {len(interactions)} interactions "
          f"across {NUM_USERS} users over {NUM_DAYS} days.")
