from typing import List, Dict, Any
from pathlib import Path
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from backend.config import CHROMA_DIR, COLLECTION_NAME, RETRIEVAL_TOP_K
from backend.services.embedding_service import EmbeddingService
from backend.utils.helpers import logger


class VectorService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.vector_store = Chroma(
            client=self.persistent_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_service.embeddings
        )

    def add_documents(self, chunks: List[Document]) -> int:
        """
        Stores document text chunks, section metadata, and HuggingFace embeddings in ChromaDB.
        Returns total chunks added.
        """
        if not chunks:
            return 0
        
        logger.info(f"Adding {len(chunks)} section-aware chunks to ChromaDB collection '{COLLECTION_NAME}'...")
        self.vector_store.add_documents(chunks)
        logger.info("Chunks successfully indexed into ChromaDB with section metadata.")
        return len(chunks)

    def similarity_search(self, query: str, document_id: str, top_k: int = RETRIEVAL_TOP_K) -> List[Document]:
        """
        Retrieves top_k high-recall candidate chunks from ChromaDB strictly filtered by document_id.
        Default top_k is set to RETRIEVAL_TOP_K (10) for V2 high-recall candidate retrieval.
        """
        logger.info(f"High-recall search in ChromaDB for doc_id '{document_id}' | Query: '{query}' | (top_k={top_k})")
        filter_dict = {"document_id": document_id}
        
        results = self.vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=filter_dict
        )
        
        logger.info(f"Retrieved {len(results)} candidate chunks from ChromaDB for document_id={document_id}")
        return results
