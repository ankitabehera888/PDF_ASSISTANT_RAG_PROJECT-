from typing import List
from backend.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION
from backend.utils.helpers import logger
import sentence_transformers

class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        logger.info(f"Initializing Hugging Face embedding model: {EMBEDDING_MODEL_NAME}...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Hugging Face embedding model loaded successfully (Dimension: {EMBEDDING_DIMENSION}).")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate 384-dimensional embeddings for a list of text chunks."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate a 384-dimensional embedding vector for a user search query."""
        return self.embeddings.embed_query(text)
