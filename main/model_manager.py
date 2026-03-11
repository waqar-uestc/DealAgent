"""
Centralized model management to avoid duplicate model loading.
All models are lazy-loaded on first use.
"""
from sentence_transformers import SentenceTransformer


class ModelManager:
    """Singleton pattern for managing ML models."""
    
    _sentence_model = None
    
    @classmethod
    def get_sentence_model(cls) -> SentenceTransformer:
        """
        Get or initialize the sentence transformer model.
        
        Returns:
            SentenceTransformer: The all-MiniLM-L6-v2 model instance
        """
        if cls._sentence_model is None:
            cls._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._sentence_model

