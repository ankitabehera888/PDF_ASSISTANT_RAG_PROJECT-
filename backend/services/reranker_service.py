from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from backend.config import RERANKER_ENABLED, RERANKER_MODEL_NAME, RERANK_TOP_K
from backend.utils.helpers import logger


class RerankerService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.enabled = RERANKER_ENABLED
        self.model_name = RERANKER_MODEL_NAME
        self.model = None

        if not self.enabled:
            logger.info("Reranker is disabled by configuration (RERANKER_ENABLED=false).")
            return

        try:
            logger.info(f"Initializing Hugging Face CrossEncoder model: '{self.model_name}'...")
            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info(f"CrossEncoder reranker '{self.model_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model '{self.model_name}': {e}. Falling back to vector search order.")
            self.enabled = False

    def rerank(
        self,
        query: str,
        candidates: List[Document],
        top_k: int = RERANK_TOP_K
    ) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Reranks candidate documents using CrossEncoder.
        Returns (ranked_documents: List[Document], debug_scores: List[dict])
        """
        if not candidates:
            return [], []

        # Fallback if disabled or failed to initialize
        if not self.enabled or self.model is None:
            logger.warning("Reranker disabled or unavailable. Returning top vector search candidates.")
            fallback_candidates = candidates[:top_k]
            fallback_scores = [
                {
                    "rank": i + 1,
                    "section": doc.metadata.get("section", "General"),
                    "page": doc.metadata.get("page", 1),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "score": 1.0 - (i * 0.05)  # Synthetic vector rank placeholder
                }
                for i, doc in enumerate(fallback_candidates)
            ]
            return fallback_candidates, fallback_scores

        logger.info(f"Reranking {len(candidates)} candidate chunks for query: '{query}' (Target top_k={top_k})...")

        # Build query-passage pairs for CrossEncoder
        # Use raw text or chunk content
        pairs = [
            (query, doc.metadata.get("raw_text", doc.page_content))
            for doc in candidates
        ]

        try:
            raw_scores = self.model.predict(pairs)
            # Standardize scores to list of floats
            scores = [float(s) for s in raw_scores]

            # Pair documents with scores and original index
            scored_docs = []
            for idx, (doc, score) in enumerate(zip(candidates, scores)):
                scored_docs.append({
                    "doc": doc,
                    "score": score,
                    "initial_rank": idx + 1
                })

            # Sort by descending score
            scored_docs.sort(key=lambda item: item["score"], reverse=True)

            ranked_candidates = [item["doc"] for item in scored_docs[:top_k]]
            debug_scores = [
                {
                    "final_rank": rank_idx + 1,
                    "initial_rank": item["initial_rank"],
                    "score": round(item["score"], 4),
                    "section": item["doc"].metadata.get("section", "General"),
                    "page": item["doc"].metadata.get("page", 1),
                    "chunk_id": item["doc"].metadata.get("chunk_id", "")
                }
                for rank_idx, item in enumerate(scored_docs[:top_k])
            ]

            logger.info(f"Reranking complete. Selected top {len(ranked_candidates)} chunks.")
            return ranked_candidates, debug_scores

        except Exception as e:
            logger.error(f"Error during reranking prediction: {e}. Returning initial candidate ordering.")
            fallback_candidates = candidates[:top_k]
            return fallback_candidates, []
