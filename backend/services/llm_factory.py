from typing import Type, TypeVar, Optional, Any, AsyncGenerator, Dict, Tuple
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import HTTPException

from backend.config import GEMINI_API_KEY, PRIMARY_LLM_MODEL, FALLBACK_LLM_MODEL
from backend.utils.helpers import logger

T = TypeVar("T", bound=BaseModel)

class LLMFactoryService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.primary_model_name = PRIMARY_LLM_MODEL
        self.fallback_model_name = FALLBACK_LLM_MODEL
        # Cache LLM instances by (model_name, temperature) to eliminate repeated client creation latency
        self._llm_cache: Dict[Tuple[str, float], ChatGoogleGenerativeAI] = {}

    def _create_llm_instance(self, model_name: str, temperature: float = 0.1) -> ChatGoogleGenerativeAI:
        """Helper to create or retrieve a cached ChatGoogleGenerativeAI instance."""
        if not self.api_key or self.api_key == "your_api_key_here":
            raise HTTPException(
                status_code=400,
                detail="GEMINI_API_KEY is not configured in backend/.env. Please configure your API key."
            )

        cache_key = (model_name, round(temperature, 2))
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        instance = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature,
            max_retries=1
        )
        self._llm_cache[cache_key] = instance
        return instance

    def _extract_text_from_content(self, content: Any) -> str:
        """Safely extract plain text from string or structured content block lists (e.g. thinking tags)."""
        if not content:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    # Filter out internal 'thinking' reasoning blocks and collect 'text'
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif "text" in part and part.get("type") != "thinking":
                        text_parts.append(part.get("text", ""))
            return "\n".join(text_parts).strip()
        return str(content).strip()

    def _is_recoverable_error(self, exc: Exception) -> bool:
        """
        Distinguishes between recoverable errors (model 404, model not found, invalid model name)
        and non-recoverable errors (invalid API key, auth failure, quota exhaustion).
        """
        err_msg = str(exc).lower()

        # Non-recoverable Auth / Key errors
        if "api_key_invalid" in err_msg or "invalid api key" in err_msg or "unauthenticated" in err_msg:
            logger.error(f"Non-recoverable authentication error: {exc}")
            return False

        # Quota / Rate limit errors
        if "resource_exhausted" in err_msg or "quota" in err_msg:
            logger.error(f"Quota/Rate-limit failure encountered: {exc}")
            return True  # Try fallback in case fallback model has different quota bucket

        # Model not found / unavailable / 404 / parsing error -> Recoverable via fallback
        return True

    def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Generates text response using PRIMARY_LLM_MODEL with automatic fallback to FALLBACK_LLM_MODEL.
        """
        # Attempt 1: Primary Model
        logger.info(f"Invoking Primary LLM model: '{self.primary_model_name}'")
        try:
            llm = self._create_llm_instance(self.primary_model_name, temperature=temperature)
            response = llm.invoke(prompt)
            content = self._extract_text_from_content(response.content)
            logger.info(f"Primary model '{self.primary_model_name}' succeeded.")
            return content
        except Exception as e:
            logger.warning(f"Primary model failed: '{self.primary_model_name}' | Reason: {e}")
            if not self._is_recoverable_error(e):
                raise HTTPException(status_code=400, detail="Invalid API key or authentication error. Please check your GEMINI_API_KEY.")

        # Attempt 2: Fallback Model
        logger.info(f"Trying fallback model: '{self.fallback_model_name}'")
        try:
            fallback_llm = self._create_llm_instance(self.fallback_model_name, temperature=temperature)
            response = fallback_llm.invoke(prompt)
            content = self._extract_text_from_content(response.content)
            logger.info(f"Fallback model '{self.fallback_model_name}' succeeded.")
            return content
        except Exception as fallback_err:
            logger.error(f"Fallback model failed: '{self.fallback_model_name}' | Reason: {fallback_err}")
            raise HTTPException(
                status_code=500,
                detail="The AI service is temporarily unavailable. Please try again."
            )

    async def generate_text_stream(self, prompt: str, temperature: float = 0.1) -> AsyncGenerator[str, None]:
        """
        Streams text response token-by-token using PRIMARY_LLM_MODEL.
        Falls back to FALLBACK_LLM_MODEL if the primary model initialization fails.
        Uses ChatGoogleGenerativeAI.astream() for true LLM-level token streaming.
        """
        # Determine which model to use
        model_name = self.primary_model_name
        try:
            llm = self._create_llm_instance(model_name, temperature=temperature)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Primary model instantiation failed for streaming: {e}")
            if not self._is_recoverable_error(e):
                raise HTTPException(status_code=400, detail="Invalid API key or authentication error.")
            model_name = self.fallback_model_name
            try:
                llm = self._create_llm_instance(model_name, temperature=temperature)
            except Exception as fallback_err:
                logger.error(f"Fallback model instantiation failed: {fallback_err}")
                raise HTTPException(status_code=500, detail="AI service temporarily unavailable.")

        logger.info(f"Starting streaming generation with model: '{model_name}'")
        try:
            async for chunk in llm.astream(prompt):
                content = self._extract_text_from_content(chunk.content)
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Streaming failed mid-generation with model '{model_name}': {e}")
            # Try fallback if primary failed mid-stream
            if model_name == self.primary_model_name and self._is_recoverable_error(e):
                logger.info(f"Attempting streaming fallback with '{self.fallback_model_name}'")
                try:
                    fallback_llm = self._create_llm_instance(self.fallback_model_name, temperature=temperature)
                    async for chunk in fallback_llm.astream(prompt):
                        content = self._extract_text_from_content(chunk.content)
                        if content:
                            yield content
                    return
                except Exception as fe:
                    logger.error(f"Fallback streaming also failed: {fe}")
            raise

    def generate_structured(self, prompt: str, schema_class: Type[T], temperature: float = 0.2) -> T:
        """
        Generates structured Pydantic output using PRIMARY_LLM_MODEL with automatic fallback to FALLBACK_LLM_MODEL.
        """
        # Attempt 1: Primary Model
        logger.info(f"Invoking Primary LLM model for structured output: '{self.primary_model_name}'")
        try:
            llm = self._create_llm_instance(self.primary_model_name, temperature=temperature)
            structured_llm = llm.with_structured_output(schema_class)
            result = structured_llm.invoke(prompt)
            if result:
                logger.info(f"Primary model '{self.primary_model_name}' structured output succeeded.")
                return result
        except Exception as e:
            logger.warning(f"Primary model structured output failed: '{self.primary_model_name}' | Reason: {e}")
            if not self._is_recoverable_error(e):
                raise HTTPException(status_code=400, detail="Invalid API key or authentication error. Please check your GEMINI_API_KEY.")

        # Attempt 2: Fallback Model
        logger.info(f"Trying fallback model for structured output: '{self.fallback_model_name}'")
        try:
            fallback_llm = self._create_llm_instance(self.fallback_model_name, temperature=temperature)
            structured_llm = fallback_llm.with_structured_output(schema_class)
            result = structured_llm.invoke(prompt)
            if result:
                logger.info(f"Fallback model '{self.fallback_model_name}' structured output succeeded.")
                return result
        except Exception as fallback_err:
            logger.error(f"Fallback model structured output failed: '{self.fallback_model_name}' | Reason: {fallback_err}")
            raise HTTPException(
                status_code=500,
                detail="The AI service is temporarily unavailable. Please try again."
            )

        raise HTTPException(
            status_code=500,
            detail="The AI service is temporarily unavailable. Please try again."
        )
