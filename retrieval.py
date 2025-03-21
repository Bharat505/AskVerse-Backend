# retrieval.py
import os
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Path to your ChromaDB folder, built by your indexing script
CHROMA_DB_PATH = "./chroma_db"
load_dotenv()  # Load environment variables from .env
# Load your OpenAI API key from env (or .env), e.g. OPENAI_API_KEY
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openai_key
)

# Load your ChromaDB
vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model
)

# Retrieve stored docs so we can do BM25 on raw text
stored_docs = vector_db.get(include=["documents", "metadatas"])
bm25_corpus = [doc.lower().split() for doc in stored_docs["documents"]] if stored_docs["documents"] else []
bm25_mapping = stored_docs["metadatas"] if stored_docs["metadatas"] else []

# Initialize BM25
bm25_model = BM25Okapi(bm25_corpus) if bm25_corpus else None

def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5):
    """
    Perform a hybrid BM25 + Embedding retrieval.
    Return a list of tuples: (source, chunk_text, weighted_score).
    """

    bm25_scores, bm25_results = [], []

    # Vector Search
    vector_results = vector_db.similarity_search_with_relevance_scores(query, k=top_k)

    # BM25
    if bm25_model:
        bm25_tokens = query.lower().split()
        bm25_raw_scores = bm25_model.get_scores(bm25_tokens)
        # top K from BM25
        bm25_top_indices = np.argsort(bm25_raw_scores)[::-1][:top_k]

        for idx in bm25_top_indices:
            bm25_results.append((bm25_mapping[idx]["source"], bm25_raw_scores[idx], idx))

    # Normalize BM25
    if bm25_results:
        bm25_scores = [score for _, score, _ in bm25_results]
        if max(bm25_scores) != min(bm25_scores):
            scaler = MinMaxScaler()
            bm25_scores = scaler.fit_transform(
                np.array(bm25_scores).reshape(-1, 1)
            ).flatten()
        else:
            bm25_scores = [1.0] * len(bm25_scores)

    # Normalize Vector
    vector_scores = [score for _, score in vector_results]
    if max(vector_scores) != min(vector_scores):
        scaler = MinMaxScaler()
        vector_scores = scaler.fit_transform(
            np.array(vector_scores).reshape(-1, 1)
        ).flatten()
    else:
        vector_scores = [1.0] * len(vector_scores)

    # Combine (alpha weighting)
    final_results = []
    for i, (bm25_doc, bm25_score, bm25_idx) in enumerate(zip(bm25_results, bm25_scores, bm25_top_indices)):
        # chunk text from stored_docs
        doc_text = stored_docs["documents"][bm25_idx]
        weighted_score = (alpha * vector_scores[i]) + ((1 - alpha) * bm25_score)
        final_results.append((bm25_doc[0], doc_text, weighted_score))

    # Also incorporate the vector results
    for i, (vec_doc, vec_score) in enumerate(vector_results):
        # doc metadata source
        vec_source = vec_doc.metadata.get("source", "unknown")
        vec_text = vec_doc.page_content
        weighted_score = (alpha * vec_score) + ((1 - alpha) * bm25_scores[i])
        final_results.append((vec_source, vec_text, weighted_score))

    # Sort by weighted_score descending
    final_results = sorted(final_results, key=lambda x: x[2], reverse=True)[:top_k]
    return final_results
