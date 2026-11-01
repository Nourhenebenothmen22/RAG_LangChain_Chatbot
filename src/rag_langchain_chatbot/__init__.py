"""Production-oriented Hybrid RAG pipeline using LangChain."""

from rag_langchain_chatbot.config import Settings, get_settings
from rag_langchain_chatbot.pipeline import HybridRAGPipeline, HybridRAGResponse, RetrievedChunk
from rag_langchain_chatbot.retrieve import HybridRetriever, hybrid_retrieval
from rag_langchain_chatbot.index import DocumentIndexer, index_documents
from rag_langchain_chatbot.router import (
    ConversationalQueryRewriter,
    IntentClassification,
    IntentResult,
    IntentRouter,
    QueryIntent,
    normalize_chat_history,
)

__all__ = [
    "Settings",
    "get_settings",
    "HybridRAGPipeline",
    "HybridRAGResponse",
    "RetrievedChunk",
    "HybridRetriever",
    "hybrid_retrieval",
    "DocumentIndexer",
    "index_documents",
    "IntentRouter",
    "IntentResult",
    "IntentClassification",
    "QueryIntent",
    "ConversationalQueryRewriter",
    "normalize_chat_history",
]


def main() -> None:
    """CLI entrypoint invoked via console script."""
    from main import cli
    cli()
