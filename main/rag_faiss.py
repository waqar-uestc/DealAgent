import faiss
import numpy as np
from gpt_evaluator import gpt_answer
from shared_data import (get_top_deals, get_faiss_cache, set_faiss_cache,
                        get_search_results, get_search_faiss_cache, set_search_faiss_cache,
                        get_current_search_keyword)
from model_manager import ModelManager


def _get_model():
    """Get shared sentence model from ModelManager."""
    return ModelManager.get_sentence_model()


def build_deals_index():
    """
    Builds a FAISS index using the current deals.
    Uses cache to avoid rebuilding if deals haven't changed.
    """
    # Check cache first
    cache = get_faiss_cache()
    if cache["index"] is not None and cache["texts"] is not None:
        return cache["index"], cache["texts"]
    
    top_deals = get_top_deals()
    if not top_deals:
        return None, None

    texts = [f"{deal['title']}. {deal.get('description', '')}" for deal in top_deals]
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Cache the result
    set_faiss_cache(index, texts)

    return index, texts


def query_deals_rag(question: str, k: int = None) -> str:
    """
    Queries the current deals using FAISS and returns a GPT-generated answer.
    
    Args:
        question: User's question
        k: Number of relevant results to retrieve (default: from Config.RAG_TOP_K)
    
    Returns:
        str: Answer based on current deals
    """
    from config import Config
    
    if k is None:
        k = Config.RAG_TOP_K
    
    index, texts = build_deals_index()
    if index is None:
        return "⚠️ No deals available to answer your question. Please fetch deals first."

    model = _get_model()
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), min(k, len(texts)))

    relevant_chunks = [texts[i] for i in I[0] if i < len(texts)]
    context = "\n".join(relevant_chunks)
    
    if not context:
        return "⚠️ No relevant deals found for your question."

    return gpt_answer(question, context)


def build_search_index():
    """
    Builds a FAISS index using search results.
    Uses cache to avoid rebuilding if results haven't changed.
    """
    cache = get_search_faiss_cache()
    if cache["index"] is not None and cache["texts"] is not None:
        return cache["index"], cache["texts"]
    
    search_results = get_search_results()
    if not search_results:
        return None, None

    texts = [f"{deal['title']}. {deal.get('summary', '')}" for deal in search_results]
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Cache the result
    set_search_faiss_cache(index, texts)

    return index, texts


def query_search_rag(question: str, k: int = 5) -> str:
    """
    Queries the search results using FAISS and returns a GPT-generated answer.
    
    Args:
        question: User's question
        k: Number of relevant results to retrieve (default: 5)
    
    Returns:
        str: Answer based on search results
    """
    index, texts = build_search_index()
    if index is None:
        keyword = get_current_search_keyword()
        if keyword:
            return f"⚠️ No search results found for '{keyword}'. Please search first."
        return "⚠️ No search results available. Please search for a product first."

    model = _get_model()
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), min(k, len(texts)))

    relevant_chunks = [texts[i] for i in I[0] if i < len(texts)]
    context = "\n".join(relevant_chunks)
    
    search_results = get_search_results()
    keyword = get_current_search_keyword()
    
    # Add search context to the question
    enhanced_question = f"Based on the following search results for '{keyword}':\n\n{context}\n\nQuestion: {question}"
    
    return gpt_answer(enhanced_question, context)