top_deals = []
llm_provider = "openai"  # default provider

# RAG index cache
_faiss_cache = {
    "index": None,
    "texts": None,
    "version": 0,
}

# Search results storage
_search_results = []
_current_search_keyword = ""
_search_faiss_cache = {
    "index": None,
    "texts": None,
    "version": 0,
}


def set_top_deals(deals):
    global top_deals, _faiss_cache
    top_deals = deals or []
    # Invalidate cache when deals change
    _faiss_cache["version"] += 1
    _faiss_cache["index"] = None
    _faiss_cache["texts"] = None


def get_top_deals():
    return top_deals


def get_faiss_cache():
    """Get the current FAISS index cache."""
    return _faiss_cache


def set_faiss_cache(index, texts):
    """Set the FAISS index cache."""
    global _faiss_cache
    _faiss_cache["index"] = index
    _faiss_cache["texts"] = texts


def set_llm_provider(provider: str):
    """Set the global LLM provider ("openai", "gemini", or "deepseek")."""
    global llm_provider
    p = (provider or "").strip().lower()
    if p not in ("openai", "gemini", "deepseek"):
        p = "openai"
    llm_provider = p


def get_llm_provider() -> str:
    """Get current LLM provider selection."""
    return llm_provider


def append_log(message: str):
    try:
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"Log write error: {e}")


def set_search_results(keyword: str, deals: list):
    """Set search results for a specific keyword."""
    global _search_results, _current_search_keyword, _search_faiss_cache
    _current_search_keyword = keyword.strip()
    _search_results = deals or []
    # Invalidate search cache when results change
    _search_faiss_cache["version"] += 1
    _search_faiss_cache["index"] = None
    _search_faiss_cache["texts"] = None


def get_search_results():
    """Get current search results."""
    return _search_results


def get_current_search_keyword():
    """Get current search keyword."""
    return _current_search_keyword


def get_search_faiss_cache():
    """Get the search results FAISS index cache."""
    return _search_faiss_cache


def set_search_faiss_cache(index, texts):
    """Set the search results FAISS index cache."""
    global _search_faiss_cache
    _search_faiss_cache["index"] = index
    _search_faiss_cache["texts"] = texts