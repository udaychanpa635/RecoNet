"""
Explainability layer: for a given recommendation, retrieves the user's most
similar past-liked item (via a LangChain FAISS vector store over item
descriptions) and surfaces it as the reason — "recommended because you
liked X" — instead of shipping a black-box ranked list.

Uses LangChain's VectorStore/Retriever abstractions so the retrieval piece
is swappable (FAISS here, could be Pinecone/Chroma/etc. with no other code
changes) and so an LLM can optionally turn the retrieved evidence into a
natural-language explanation.

Two modes:
  - Template mode (default, no API key needed): fills a plain-English
    template from the retrieved evidence. Fully testable offline.
  - LLM mode (set ANTHROPIC_API_KEY): passes the same retrieved evidence to
    Claude via langchain-anthropic for a more natural explanation sentence.

    python src/explainer.py
"""
import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from data_utils import load_data


class HashedBOWEmbeddings(Embeddings):
    """
    Same lightweight placeholder used in pytorch_two_tower.py, exposed
    through LangChain's Embeddings interface so it can back a FAISS
    VectorStore. Swap for langchain_huggingface.HuggingFaceEmbeddings with a
    real sentence-transformer model once you have Hugging Face Hub access.
    """
    def __init__(self, dim: int = 32):
        self.dim = dim

    def _embed(self, text: str):
        import numpy as np
        vec = np.zeros(self.dim, dtype="float32")
        for word in text.lower().replace(",", "").replace(".", "").split():
            vec[hash(word) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def build_item_vectorstore(items) -> FAISS:
    docs = [
        Document(
            page_content=item["description"],
            metadata={"item_id": item["item_id"], "name": item["name"], "category": item["category"]},
        )
        for item in items
    ]
    return FAISS.from_documents(docs, HashedBOWEmbeddings())


def explain_recommendation(
    vectorstore: FAISS,
    recommended_item: dict,
    user_liked_items: list,
    use_llm: bool = False,
) -> str:
    """
    Finds which of the user's previously-liked items is most similar (by
    content) to the newly recommended item, and explains the recommendation
    in terms of it.
    """
    if not user_liked_items:
        return f"Recommended '{recommended_item['name']}' based on overall popularity — this user has no history yet."

    retriever = vectorstore.as_retriever(search_kwargs={"k": len(user_liked_items) + 1})
    results = retriever.invoke(recommended_item["description"])

    liked_ids = {item["item_id"] for item in user_liked_items}
    best_match = None
    for doc in results:
        if doc.metadata["item_id"] in liked_ids and doc.metadata["item_id"] != recommended_item["item_id"]:
            best_match = doc.metadata
            break

    if best_match is None:
        return f"Recommended '{recommended_item['name']}' based on overall trends in the {recommended_item['category']} category."

    evidence = (
        f"User previously liked '{best_match['name']}' ({best_match['category']}). "
        f"Recommended item is '{recommended_item['name']}' ({recommended_item['category']}), "
        f"which is content-similar."
    )

    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        return _explain_with_llm(evidence, recommended_item, best_match)

    return f"Recommended because you liked '{best_match['name']}' — both are similar {recommended_item['category']} items."


def _explain_with_llm(evidence: str, recommended_item: dict, best_match: dict) -> str:
    """Optional: use Claude via langchain-anthropic to phrase the explanation naturally."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=100)
    prompt = ChatPromptTemplate.from_template(
        "Write one short, friendly sentence explaining a product recommendation "
        "to a customer, grounded ONLY in this evidence (don't invent anything "
        "beyond it):\n\n{evidence}\n\nExplanation:"
    )
    chain = prompt | llm
    response = chain.invoke({"evidence": evidence})
    return response.content


if __name__ == "__main__":
    items, interactions = load_data()
    vectorstore = build_item_vectorstore(items)

    # demo: a user who has liked a couple of fitness items, get an explanation
    # for a new fitness recommendation
    fitness_items = [i for i in items if i["category"] == "fitness"]
    user_liked_items = fitness_items[:2]
    recommended_item = fitness_items[2]

    print(f"User's liked items: {[i['name'] for i in user_liked_items]}")
    print(f"Recommended item: {recommended_item['name']}")
    print(f"\nExplanation (template mode):")
    print(" ", explain_recommendation(vectorstore, recommended_item, user_liked_items, use_llm=False))

    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"\nExplanation (LLM mode via Claude):")
        print(" ", explain_recommendation(vectorstore, recommended_item, user_liked_items, use_llm=True))
    else:
        print("\n(Set ANTHROPIC_API_KEY to also see the LLM-phrased explanation.)")
