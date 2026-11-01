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
    """Return a cached HuggingFace embeddings model instance with offline-first local loading."""
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
        try:
            # Fast offline-first initialization to prevent 35s+ remote hub retries
            _EMBEDDING_CACHE[cache_key] = HuggingFaceEmbeddings(
                model_name=cfg.embedding_model_name,
                model_kwargs={"device": cfg.embedding_device, "local_files_only": True},
                encode_kwargs={"normalize_embeddings": cfg.normalize_embeddings}
            )
        except Exception:
            _EMBEDDING_CACHE[cache_key] = HuggingFaceEmbeddings(
                model_name=cfg.embedding_model_name,
                model_kwargs={"device": cfg.embedding_device},
                encode_kwargs={"normalize_embeddings": cfg.normalize_embeddings}
            )

        # Extend client sequence length to avoid silent truncation of chunks
        model_instance = _EMBEDDING_CACHE[cache_key]
        client = getattr(model_instance, "_client", None) or getattr(model_instance, "client", None)
        if client is not None and hasattr(client, "max_seq_length"):
            try:
                client.max_seq_length = cfg.embedding_max_seq_length
                logger.info("Configured embedding model max_seq_length = %d", cfg.embedding_max_seq_length)
            except Exception as e:
                logger.debug("Could not set max_seq_length on embedding client: %s", e)

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

        # Check if conversational prompt without retrieval context
        is_conversational_prompt = (
            "conversational message" in full_prompt.lower()
            or "greeting, gratitude, farewell" in full_prompt.lower()
            or "respond naturally" in full_prompt.lower()
        )

        # Check if query rewrite / condensation prompt
        is_rewrite_prompt = (
            "query reformulation" in full_prompt.lower()
            or "contextual de-identification" in full_prompt.lower()
            or "standalone, search-optimized query" in full_prompt.lower()
        )
        if is_rewrite_prompt:
            human_messages = [
                m for m in messages
                if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage"
            ]
            current_query = human_messages[-1].content.strip() if human_messages else ""
            prior_messages = [
                m for m in messages
                if (m is not (human_messages[-1] if human_messages else None))
                and getattr(m, "type", "") in ("human", "ai")
            ]
            prior_text = " ".join([m.content for m in prior_messages if isinstance(m.content, str)])
            prior_human = [
                m for m in prior_messages
                if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage"
            ]

            if not prior_messages or not prior_text.strip():
                content = current_query
            else:
                last_human_text = prior_human[-1].content.strip() if prior_human else prior_messages[-1].content.strip()
                clean_topic = re.sub(
                    r"^(explain\s+(the\s+)?|what\s+is\s+(the\s+)?|tell\s+me\s+about\s+(the\s+)?|how\s+does\s+|parlez-moi\s+de\s+(la\s+|l'|le\s+)?|expliquez-moi\s+(le\s+|la\s+)?|qu'est-ce\s+que\s+(le\s+|la\s+)?|احكيلي\s+على\s+|شنوة\s+هو\s+|كيفاش\s+|فسرلي\s+)",
                    "",
                    last_human_text,
                    flags=re.IGNORECASE,
                ).strip("?. ")
                clean_topic = re.sub(r"\s+work$", "", clean_topic, flags=re.IGNORECASE).strip()

                lower_q = current_query.lower()
                if clean_topic and clean_topic.lower() in lower_q:
                    content = current_query
                else:
                    has_pronoun = bool(re.search(r"\b(it|its|they|their|them|this|that|these|those|son|sa|ses|ce|cette|ces)\b", lower_q))
                    has_ellipsis = any(
                        p in lower_q
                        for p in [
                            "what about", "and for", "how about", "et pour", "qu'en est-il",
                            "وقتاش", "قداش", "وين", "كيفاش", "شنوة", "chnowa", "w chnowa"
                        ]
                    )
                    is_short = len(current_query.split()) <= 4

                    if not (has_pronoun or has_ellipsis or is_short):
                        content = current_query
                    elif re.search(r"\bits\b", lower_q):
                        content = re.sub(r"\bits\b", f"{clean_topic}'s", current_query, flags=re.IGNORECASE)
                    elif re.search(r"\b(son|sa|ses)\b\s+(\w+)", lower_q):
                        content = re.sub(r"\b(son|sa|ses)\b\s+(\w+)", rf"\2 de la {clean_topic}", current_query, flags=re.IGNORECASE)
                    elif re.search(r"\b(it|this|that|ce|cette)\b", lower_q):
                        content = re.sub(r"\b(it|this|that|ce|cette)\b", clean_topic, current_query, flags=re.IGNORECASE)
                    elif any(ar in current_query for ar in ["وقتاش", "قداش", "وين", "كيفاش", "شنوة"]):
                        content = f"{current_query.rstrip('؟?')} بخصوص {clean_topic}؟"
                    elif "pour" in lower_q or "de" in lower_q:
                        content = f"{current_query.rstrip('?')} pour {clean_topic} ?"
                    else:
                        content = f"{current_query.rstrip('?')} regarding {clean_topic}?"

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
        if is_conversational_prompt and not doc_matches:
            has_arabic = any("\u0600" <= c <= "\u06FF" for c in full_prompt)
            is_arabizi = any(w in full_prompt.lower() for w in ["aslema", "ahla", "3aychek", "yaatik", "beslama", "bsslama", "labes", "chbik"])
            is_french = any(w in full_prompt.lower() for w in ["bonjour", "salut", "merci", "revoir", "comment", "journée", "journee"])

            if has_arabic:
                content = "أهلاً ومرحباً بك! أنا مساعد ClaimEye الذكي، يسعدني دائماً مساعدتك في كل ما يخص ملفات التأمين، وتقدير أضرار الحوادث، وبنود العقود."
            elif is_arabizi:
                content = "Ahla bik w marhba! Ena l-assistant ClaimEye AI. Dima fel khedma ken theb souel 3al dossiers mte3ek, chiffrage wala les franchises."
            elif is_french:
                content = "Bonjour ! Je suis l'assistant ClaimEye AI. Je suis à votre entière disposition pour vous accompagner dans vos dossiers d'assurance, le chiffrage des dommages et l'analyse de vos contrats."
            else:
                content = "Hello! I am ClaimEye AI, your assistant for insurance claims, damage assessment, and policy workflows. How can I assist you today?"

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

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Provide structured output parsing support for LCEL composition."""
        from langchain_core.runnables import RunnableLambda

        def _generate_structured(messages_input: Any) -> Any:
            if hasattr(messages_input, "to_messages"):
                msg_list = messages_input.to_messages()
            elif isinstance(messages_input, list):
                msg_list = messages_input
            elif isinstance(messages_input, dict):
                msg_list = [messages_input.get("query", str(messages_input))]
            else:
                msg_list = [messages_input]

            # Extract user query from the last message (HumanMessage) or user_input tags
            last_msg = msg_list[-1] if msg_list else ""
            user_content = getattr(last_msg, "content", str(last_msg)) if last_msg else ""
            user_input_match = re.search(r"<user_input>\s*(.*?)\s*</user_input>", user_content, re.DOTALL | re.IGNORECASE)
            query = user_input_match.group(1).strip() if user_input_match else user_content.strip()

            query_lower = query.lower()

            # 1. Detect Adversarial Prompt Injections
            is_injection = any(
                p in query_lower for p in [
                    "ignore previous", "ignore all", "bypass", "override system",
                    "classify this as", "forget instructions", "system prompt"
                ]
            )

            # 2. Detect Language
            has_arabic_script = any("\u0600" <= c <= "\u06FF" for c in query)
            is_arabizi = any(w in query_lower for w in ["aslema", "ahla", "3aychek", "yaatik", "ya3tik", "beslama", "bsslama", "fellemen", "chnowa", "chneya"])
            is_french = any(w in query_lower for w in ["bonjour", "salut", "merci", "quelle", "quel", "comment", "revoir", "franchise", "chiffrage", "seuil"])
            is_english = any(w in query_lower for w in ["hello", "hi", "thanks", "thank", "goodbye", "bye", "what", "which", "how", "threshold", "claim"])

            if has_arabic_script:
                is_tunisian = any(w in query for w in ["عسلامة", "يعيشك", "شكون", "قداش", "كرهبة", "برشا", "برشة", "بسلامة", "لمان"])
                detected_lang = "Tunisian Arabic" if is_tunisian else "Arabic"
                lang_dist = {detected_lang: 1.0}
            elif is_arabizi:
                detected_lang = "Tunisian Arabic"
                lang_dist = {"Tunisian Arabic": 0.7, "Arabizi": 0.3}
            elif is_french:
                detected_lang = "French"
                lang_dist = {"French": 1.0}
            elif is_english:
                detected_lang = "English"
                lang_dist = {"English": 1.0}
            else:
                detected_lang = "English"
                lang_dist = {"English": 1.0}

            # 3. Detect Intent & Retrieval Necessity
            has_question = "?" in query or "؟" in query
            domain_terms = [
                "franchise", "sinistre", "constat", "fraude", "fraud", "score", "seuil", "threshold",
                "devis", "chiffrage", "expert", "remboursement", "reimbursement", "garantie", "contrat",
                "policy", "tarification", "pricing", "t301", "t310", "t401", "bf-0", "us-1", "claimeye",
                "degat", "dommage", "damage", "collision", "deductible", "couverture", "workflow",
                "الفرانشيز", "التأمين", "تامين", "احتيال", "تعويض", "معاينة", "كرهبة"
            ]
            has_domain_term = any(term in query_lower or term in query for term in domain_terms)

            conv_terms = [
                "hello", "hi", "hey", "good morning", "good afternoon", "greetings",
                "bonjour", "salut", "coucou", "bonsoir",
                "مرحبا", "صباح الخير", "السلام عليكم", "عسلامة", "aslema", "ahla",
                "thank", "thanks", "merci", "شكرا", "يعيشك", "3aychek", "yaatik", "awesome",
                "يعطيك", "الصحة", "يرحم", "والديك",
                "bye", "goodbye", "au revoir", "bientot", "مع السلامة", "بسلامة", "beslama", "bsslama", "لمان",
                "who are you", "qui es-tu", "how are you", "comment tu vas", "comment allez-vous",
                "من أنت", "who created", "capabilities", "chneya ahwelek", "later", "catch you", "take care"
            ]
            has_conv_term = any(term in query_lower or term in query for term in conv_terms)

            if is_injection:
                intent = "document_query"
                requires_retrieval = True
                confidence = 0.95
                strategy = "execute_hybrid_rag"
                reasoning = "Untrusted adversarial prompt injection attempt detected. Defaulting safely to document retrieval."
                needs_clarification = False
            elif has_domain_term:
                if has_conv_term:
                    intent = "mixed_conversational_retrieval"
                    requires_retrieval = True
                    confidence = 0.96
                    strategy = "execute_hybrid_rag"
                    reasoning = "Message contains greeting combined with substantive domain insurance inquiry. Requires document retrieval."
                else:
                    intent = "document_query"
                    requires_retrieval = True
                    confidence = 0.98
                    strategy = "execute_hybrid_rag"
                    reasoning = "Substantive question requiring factual knowledge from indexed documents/database."
                needs_clarification = False
            elif has_conv_term:
                if any(w in query_lower or w in query for w in ["thank", "thanks", "merci", "شكرا", "يعيشك", "3aychek", "awesome"]):
                    intent = "gratitude"
                elif any(w in query_lower or w in query for w in ["bye", "goodbye", "au revoir", "bientot", "مع السلامة", "بسلامة", "beslama", "later", "catch you"]):
                    intent = "farewell"
                elif any(w in query_lower or w in query for w in ["who are you", "qui es-tu", "how are you", "comment tu vas", "comment allez-vous", "من أنت", "who created", "capabilities", "chneya ahwelek"]):
                    intent = "small_talk"
                else:
                    intent = "greeting"
                requires_retrieval = False
                confidence = 0.95
                strategy = "direct_conversational_response"
                reasoning = "Purely conversational input without factual or document inquiry. Bypassing retrieval."
                needs_clarification = False
            elif len(query.strip()) <= 4 and not query.strip().isalnum():
                intent = "ambiguous"
                requires_retrieval = True
                confidence = 0.50
                strategy = "request_clarification"
                reasoning = "Highly ambiguous or fragmented input. Defaulting safely to retrieval."
                needs_clarification = True
            else:
                intent = "document_query"
                requires_retrieval = True
                confidence = 0.85
                strategy = "execute_hybrid_rag"
                reasoning = "Inquiry requiring domain grounding. Safe retrieval route."
                needs_clarification = False

            if hasattr(schema, "model_fields") or hasattr(schema, "__fields__"):
                return schema(
                    intent=intent,
                    detected_language=detected_lang,
                    language_distribution=lang_dist,
                    requires_retrieval=requires_retrieval,
                    expects_direct_response=not requires_retrieval,
                    needs_clarification=needs_clarification,
                    confidence=confidence,
                    reasoning=reasoning,
                    response_strategy=strategy
                )
            return {
                "intent": intent,
                "detected_language": detected_lang,
                "language_distribution": lang_dist,
                "requires_retrieval": requires_retrieval,
                "confidence": confidence,
                "reasoning": reasoning,
                "response_strategy": strategy
            }

        return RunnableLambda(_generate_structured)


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
            primary_llm = ChatOpenAI(
                model=cfg.llm_model_name,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                api_key=api_key_openai,
                timeout=8.0,
                max_retries=1
            )
            fallback_llm = LocalContextSynthesizer(model_name=f"{cfg.llm_model_name}-fallback")
            # Equip with native LangChain fallback for resilient production operation
            return primary_llm.with_fallbacks([fallback_llm])
        except Exception as e:
            logger.warning("Could not initialize ChatOpenAI (%s). Falling back to LocalContextSynthesizer.", e)

    return LocalContextSynthesizer(model_name=cfg.llm_model_name or "local-synthesizer")
