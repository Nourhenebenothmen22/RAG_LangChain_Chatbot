"""Tests for native LangChain BaseRetriever compliance and LCEL runnable pipelines."""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rag_langchain_chatbot.pipeline import HybridRAGPipeline
from rag_langchain_chatbot.retrieve import HybridRetriever


def test_hybrid_retriever_is_langchain_base_retriever():
    """Verify HybridRetriever is an instance of LangChain BaseRetriever."""
    retriever = HybridRetriever()
    assert isinstance(retriever, BaseRetriever)


def test_hybrid_retriever_invoke_interface():
    """Verify retriever.invoke() returns a list of Document instances."""
    retriever = HybridRetriever()
    docs = retriever.invoke("What happens when the fraud score exceeds 0.6?")
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    assert "rerank_score" in docs[0].metadata


def test_lcel_chain_execution():
    """Verify pipeline.chain executes as an idiomatic LCEL chain (retriever | prompt | llm)."""
    pipeline = HybridRAGPipeline()
    chain = pipeline.chain
    
    # Execute LCEL chain
    result = chain.invoke("What happens when the fraud score exceeds 0.6?")
    assert isinstance(result, str)
    assert len(result) > 20
    assert ("0.6" in result or "0,6" in result or "expert" in result.lower())
