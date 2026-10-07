# RecoNet — Hybrid Recommender System with Explainability

Recommends items to users by combining three approaches — a popularity
baseline, a **TensorFlow** neural matrix-factorization model, and a
**PyTorch** two-tower model that fuses collaborative + content signals —
then explains each recommendation using a **LangChain** retrieval layer
("recommended because you liked X") instead of shipping a black-box list.

Built to add a second, distinct project alongside a computer-vision fine-
tuning project on the same resume — this one demonstrates ranking metrics,
embeddings, and multi-framework ML engineering instead of a second
vision/document task.

## What's tested and working right now (no GPU needed)

| Component | Status |
|---|---|
| Synthetic data generator | Tested |
| Popularity baseline + temporal split (`src/data_utils.py`) | Tested |
| TensorFlow matrix factorization (`src/tf_mf_model.py`) | Tested, trains in seconds on CPU |
| PyTorch two-tower model (`src/pytorch_two_tower.py`) | Tested, trains in seconds on CPU |
| Cold-start evaluation (`src/cold_start_eval.py`) | Tested — see honest caveat below |
| LangChain explainability layer (`src/explainer.py`) | Tested |
| Comparison script (`src/compare_all.py`) | Tested |
| FastAPI app (`api/main.py`) | Tested end-to-end over HTTP |
| Frontend demo | Working, calls the live API |

**One real engineering note:** TensorFlow and PyTorch loaded in the same
process can segfault (conflicting native runtime libraries). `compare_all.py`
runs each model in its own subprocess instead — a legitimate pattern for
multi-framework pipelines, worth mentioning if it comes up in an interview.

## Honest caveat on results

The bundled synthetic dataset (32 items, 300 users) is small, and item
"content embeddings" here are a placeholder hashed bag-of-words (no GPU/HF
Hub access in the environment this was built in). On this toy data, the
fancier two-tower model doesn't clearly beat the simpler baselines, and the
cold-start advantage doesn't show up cleanly — that's an honest result of a
weak placeholder embedding on a tiny dataset, not a hidden bug. Swap in real
sentence-transformer embeddings and a real dataset (MovieLens-25M or Amazon
Reviews — see the Colab notebook) before reporting final numbers.

## Quickstart

```bash
pip install -r requirements.txt
python sample_data/generate_data.py

# Run each model and see real metrics
python src/data_utils.py          # popularity baseline
python src/tf_mf_model.py         # TensorFlow model
python src/pytorch_two_tower.py   # PyTorch model
python src/cold_start_eval.py     # cold-start ranking check
python src/compare_all.py         # side-by-side comparison table
python src/explainer.py           # LangChain explanation demo

# Run the API + frontend
uvicorn api.main:app --reload --port 8001
# open frontend/index.html in a browser
```

## Training on a real dataset

Swap `sample_data/generate_data.py`'s output for MovieLens-25M or Amazon
Reviews, and swap `HashedBOWEmbeddings` in `src/explainer.py` /
`build_content_features` in `src/pytorch_two_tower.py` for real
sentence-transformer embeddings (`langchain_huggingface.HuggingFaceEmbeddings`
with `all-MiniLM-L6-v2` is a good default). Do this on a machine with GPU
and Hugging Face Hub access — a Colab notebook is the easiest path, same
pattern as the DocuMind project's training notebook.

## Resume bullet (fill in your real numbers after running on a real dataset)

> Built a hybrid recommender (TensorFlow matrix factorization + PyTorch
> two-tower model with content fusion) on [dataset], evaluated with
> Precision@K/Recall@K/NDCG@K on a temporal split; added a LangChain
> retrieval layer that grounds each recommendation in a specific past-liked
> item instead of a black-box score.

## Project structure

```
reconet/
├── src/
│   ├── data_utils.py         # temporal split, popularity baseline, eval metrics
│   ├── tf_mf_model.py        # TensorFlow matrix factorization
│   ├── pytorch_two_tower.py  # PyTorch two-tower model (collaborative + content)
│   ├── cold_start_eval.py    # tests the actual value of content fusion
│   ├── explainer.py          # LangChain retrieval-based explanations
│   └── compare_all.py        # side-by-side comparison (subprocess-isolated)
├── api/main.py                # FastAPI serving
├── frontend/index.html        # demo UI
├── sample_data/generate_data.py
└── requirements.txt
```
