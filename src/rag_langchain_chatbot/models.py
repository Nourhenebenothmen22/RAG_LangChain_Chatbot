"""Shared factory for local embedding models, Chroma vector store, and LLMs."""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import HuggingFaceEmbeddings

from rag_langchain_chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)

_EMBEDDING_CACHE: Dict[tuple[str, str, bool], HuggingFaceEmbeddings] = {}


def get_embedding_model(settings: Optional[Settings] = None) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embeddings model instance."""
    cfg = settings or get_settings()
    cache_key = (
        cfg.embedding_model_name,
        cfg.embedding_device,
        cfg.normalize_embeddings
    )

    if cache_key not in _EMBEDDING_CACHE:
        logger.info(
            "Initializing embedding model: %s on %s",
            cfg.embedding_model_name,
            cfg.embedding_device
        )
        _EMBEDDING_CACHE[cache_key] = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model_name,
            model_kwargs={"device": cfg.embedding_device},
            encode_kwargs={"normalize_embeddings": cfg.normalize_embeddings}
        )

    return _EMBEDDING_CACHE[cache_key]


def get_vector_store(
    settings: Optional[Settings] = None,
    embeddings: Optional[HuggingFaceEmbeddings] = None
) -> Chroma:
    """Initialize and return a persistent Chroma vector store."""
    cfg = settings or get_settings()
    embedding_fn = embeddings or get_embedding_model(cfg)

    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embedding_fn,
        persist_directory=str(cfg.chroma_dir)
    )


class LocalContextSynthesizer(BaseChatModel):
    """Generic extractive local synthesizer for grounded answer generation from retrieved context."""

    model_name: str = "local-synthesizer"

    @property
    def _llm_type(self) -> str:
        return "local-context-synthesizer"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        full_prompt = " ".join([m.content for m in messages if isinstance(m.content, str)])

        # Extract user query
        question_match = re.search(r"(?:User Question|Question):\s*(.*?)(?:\n|$)", full_prompt, re.IGNORECASE)
        question = question_match.group(1).strip() if question_match else ""

        # Extract document texts from XML blocks or prompt
        doc_matches = re.findall(r"<document[^>]*>(.*?)</document>", full_prompt, re.DOTALL | re.IGNORECASE)
        context_corpus = " ".join(doc_matches) if doc_matches else full_prompt

        # Clean and extract candidate sentences from retrieved context
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?\n])\s+", context_corpus)
            if len(s.strip()) > 15
            and not s.strip().startswith("<")
            and not s.strip().startswith("User Question")
            and not s.strip().startswith("Retrieved database evidence")
        ]

        # Tokenize query for relevance scoring
        query_words = set(re.findall(r"\w+", question.lower())) if question else set()
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "this", "that", "with",
            "from", "the", "and", "is", "are", "for", "in", "to", "how", "does", "happens"
        }
        meaningful_query_words = {w for w in query_words if len(w) > 2 and w not in stop_words}

        scored_sentences: List[tuple[int, str]] = []
        seen = set()
        for s in raw_sentences:
            s_clean = " ".join(s.split())
            if s_clean in seen:
                continue
            seen.add(s_clean)
            s_words = set(re.findall(r"\w+", s_clean.lower()))
            overlap = len(s_words & meaningful_query_words)
            if overlap > 0:
                scored_sentences.append((overlap, s_clean))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored_sentences[:4]]

        if top_sentences:
            bullets = "\n".join(f"- {s}" for s in top_sentences)
            content = f"Based on the retrieved database records and documents:\n\n{bullets}"
        elif doc_matches:
            substantive = [s for s in raw_sentences[:3] if len(s) > 20]
            if substantive:
                bullets = "\n".join(f"- {s}" for s in substantive)
                content = f"Based on the available evidence in the retrieved documents:\n\n{bullets}"
            else:
                content = "Information found in the retrieved documents, but insufficient specific details match the query terms."
        else:
            content = "Information not found in the indexed database and documents. Please verify the query or ingest the relevant sources."

        prompt_tokens = max(1, len(full_prompt) // 4)
        completion_tokens = max(1, len(content) // 4)
        total_tokens = prompt_tokens + completion_tokens

        generation = ChatGeneration(
            message=AIMessage(
                content=content,
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    "model_name": self.model_name,
                }
            )
        )
        return ChatResult(generations=[generation])


def get_llm(settings: Optional[Settings] = None) -> BaseChatModel:
    """Initialize and return the configured LLM instance with LangSmith tracing support."""
    cfg = settings or get_settings()

    # Only enable LangSmith Tracing V2 if an API key is explicitly configured
    api_key_langsmith = cfg.langchain_api_key or os.environ.get("LANGCHAIN_API_KEY")
    if cfg.langchain_tracing_v2 and api_key_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if cfg.langchain_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = cfg.langchain_endpoint
        if cfg.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = cfg.langchain_project
        os.environ["LANGCHAIN_API_KEY"] = api_key_langsmith
    else:
        # Disable cloud tracing if no valid key to avoid 401 spam
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    api_key_openai = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY")

    if cfg.llm_provider.lower() == "openai" and api_key_openai and not api_key_openai.startswith("sk-placeholder"):
        try:
            from langchain_openai import ChatOpenAI
            logger.info("Configured ChatOpenAI model: %s", cfg.llm_model_name)
            return ChatOpenAI(
                model=cfg.llm_model_name,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                api_key=api_key_openai
            )
        except Exception as e:
            logger.warning("Could not initialize ChatOpenAI (%s). Falling back to LocalContextSynthesizer.", e)

    return LocalContextSynthesizer(model_name=cfg.llm_model_name or "local-synthesizer")
