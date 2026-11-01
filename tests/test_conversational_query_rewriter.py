"""Tests for Conversational Query Rewriting with LCEL."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_langchain_chatbot.models import LocalContextSynthesizer
from rag_langchain_chatbot.pipeline import HybridRAGPipeline
from rag_langchain_chatbot.router import (
    ConversationalQueryRewriter,
    normalize_chat_history,
)


def test_normalize_chat_history_various_formats():
    """Verify chat history normalization handles dicts, tuples, and BaseMessage instances."""
    # Dict format with role
    history_dict = [
        {"role": "user", "content": "What is insurance?"},
        {"role": "assistant", "content": "Insurance is risk management."},
    ]
    normalized = normalize_chat_history(history_dict)
    assert len(normalized) == 2
    assert isinstance(normalized[0], HumanMessage)
    assert normalized[0].content == "What is insurance?"
    assert isinstance(normalized[1], AIMessage)
    assert normalized[1].content == "Insurance is risk management."

    # Dict format with type
    history_type = [
        {"type": "human", "content": "Tell me more."},
        {"type": "ai", "content": "Sure thing."},
    ]
    normalized_type = normalize_chat_history(history_type)
    assert len(normalized_type) == 2
    assert isinstance(normalized_type[0], HumanMessage)
    assert isinstance(normalized_type[1], AIMessage)

    # Tuple format
    history_tuples = [
        ("user", "Hello"),
        ("ai", "Hi there"),
        ("system", "Context"),
    ]
    normalized_tuples = normalize_chat_history(history_tuples)
    assert len(normalized_tuples) == 3
    assert isinstance(normalized_tuples[0], HumanMessage)
    assert isinstance(normalized_tuples[1], AIMessage)
    assert isinstance(normalized_tuples[2], SystemMessage)

    # Already BaseMessages
    messages = [HumanMessage(content="Q1"), AIMessage(content="A1")]
    normalized_msgs = normalize_chat_history(messages)
    assert normalized_msgs == messages

    # Empty or None
    assert normalize_chat_history(None) == []
    assert normalize_chat_history([]) == []


def test_rewriter_preserves_standalone_query_without_history():
    """Verify query rewriter returns the exact query when no history is provided."""
    rewriter = ConversationalQueryRewriter(llm=LocalContextSynthesizer())
    original = "What is the fraud detection threshold?"
    rewritten = rewriter.rewrite(query=original, chat_history=[])
    assert rewritten == original

    rewritten_none = rewriter.rewrite(query=original, chat_history=None)
    assert rewritten_none == original


def test_rewriter_preserves_already_independent_query_with_history():
    """Verify an already self-contained query is not altered even if history exists."""
    rewriter = ConversationalQueryRewriter(llm=LocalContextSynthesizer())
    history = [
        HumanMessage(content="What is third-party liability insurance?"),
        AIMessage(content="It covers damages caused to other parties."),
    ]
    independent_query = "What is the deadline to report a car accident claim?"
    rewritten = rewriter.rewrite(query=independent_query, chat_history=history)
    # The query is already standalone, so the core intent and terms must remain intact
    assert "car accident claim" in rewritten.lower() or "deadline" in rewritten.lower()


def test_rewriter_resolves_pronouns_english():
    """Verify English conversational ellipsis and pronouns are resolved into standalone query."""
    rewriter = ConversationalQueryRewriter(llm=LocalContextSynthesizer())
    history = [
        HumanMessage(content="Explain the auto claim fraud detection mechanism."),
        AIMessage(content="Auto claim fraud uses rule-based scoring and anomaly detection."),
    ]
    follow_up = "What is its threshold for manual review?"
    rewritten = rewriter.rewrite(query=follow_up, chat_history=history)
    
    assert rewritten != follow_up
    assert "fraud" in rewritten.lower() or "auto claim" in rewritten.lower()
    assert "threshold" in rewritten.lower()


def test_rewriter_resolves_pronouns_french():
    """Verify French conversational context resolution."""
    rewriter = ConversationalQueryRewriter(llm=LocalContextSynthesizer())
    history = [
        HumanMessage(content="Parlez-moi de la détection de fraude automobile."),
        AIMessage(content="Le système attribue un score de suspicion."),
    ]
    follow_up = "Quel est son seuil de déclenchement ?"
    rewritten = rewriter.rewrite(query=follow_up, chat_history=history)
    
    assert "fraude" in rewritten.lower() or "suspicion" in rewritten.lower() or "automobile" in rewritten.lower()
    assert "seuil" in rewritten.lower()


def test_rewriter_resolves_arabic_and_tunisian():
    """Verify Arabic / Tunisian dialect conversational context resolution."""
    rewriter = ConversationalQueryRewriter(llm=LocalContextSynthesizer())
    history = [
        HumanMessage(content="احكيلي على كشف الاحتيال في حوادث السيارات"),
        AIMessage(content="السيستام يحسب سكور الشبهة بناء على معايير محددة."),
    ]
    follow_up = "وقتاش يطيح في المراجعة اليدوية؟"
    rewritten = rewriter.rewrite(query=follow_up, chat_history=history)
    
    assert len(rewritten) > len(follow_up)
    assert any(term in rewritten for term in ["الاحتيال", "السيارات", "سكور", "حوادث"])


def test_pipeline_integration_with_query_rewriting():
    """Verify HybridRAGPipeline end-to-end resolves follow-ups and stores standalone_query."""
    pipeline = HybridRAGPipeline()
    history = [
        {"role": "user", "content": "How does fraud detection work?"},
        {"role": "assistant", "content": "The system computes a fraud score between 0 and 1."},
    ]
    follow_up = "What happens if it exceeds 0.6?"
    response = pipeline.query(query_text=follow_up, chat_history=history)
    
    assert response.standalone_query is not None
    assert "fraud" in response.standalone_query.lower()
    assert len(response.chunks) > 0
    # The answer should reflect the retrieved fraud knowledge
    assert response.is_grounded is True
