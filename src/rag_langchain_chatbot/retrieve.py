import hashlib
import re
import json

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CHROMA_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "claimeye_documents"

DENSE_TOP_K = 10
BM25_TOP_K = 10
RRF_K = 60
FINAL_TOP_K = 5


# --------------------------------------------------
# 2. Initialize embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# --------------------------------------------------
# 3. Connect to persistent ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIRECTORY
)


# --------------------------------------------------
# 4. Load indexed documents for BM25
# --------------------------------------------------

def load_indexed_documents():

    stored_data = vector_store.get(
        include=["documents", "metadatas"]
    )

    documents = []

    for content, metadata in zip(
        stored_data["documents"],
        stored_data["metadatas"]
    ):
        documents.append(
            Document(
                page_content=content,
                metadata=metadata or {}
            )
        )

    return documents


# --------------------------------------------------
# 5. Tokenization for BM25
# --------------------------------------------------

def tokenize(text: str):

    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE
    )


# --------------------------------------------------
# 6. Initialize BM25 retriever
# --------------------------------------------------

def initialize_bm25():

    documents = load_indexed_documents()

    if not documents:
        raise ValueError(
            "ChromaDB is empty. Run index.py before retrieval."
        )

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=tokenize
    )

    bm25_retriever.k = BM25_TOP_K

    return bm25_retriever


# --------------------------------------------------
# 7. Generate a stable document key
# --------------------------------------------------

def get_document_key(document: Document):

    metadata = json.dumps(
        document.metadata,
        sort_keys=True,
        ensure_ascii=False
    )

    content = document.page_content + metadata

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# 8. Reciprocal Rank Fusion (RRF)
# --------------------------------------------------

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=RRF_K,
    top_k=FINAL_TOP_K
):

    fused_scores = {}
    documents = {}

    ranked_lists = [
        dense_results,
        bm25_results
    ]

    for ranked_list in ranked_lists:

        seen_in_list = set()

        for rank, document in enumerate(ranked_list, start=1):

            doc_id = get_document_key(document)

            # Prevent duplicate contributions within one ranking
            if doc_id in seen_in_list:
                continue

            seen_in_list.add(doc_id)

            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                documents[doc_id] = document

            fused_scores[doc_id] += 1 / (k + rank)

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        (documents[doc_id], score)
        for doc_id, score in ranked_results[:top_k]
    ]


# --------------------------------------------------
# 9. Hybrid retrieval
# --------------------------------------------------

def hybrid_retrieval(
    query: str,
    top_k: int = FINAL_TOP_K
):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Dense retrieval
    dense_results = vector_store.similarity_search(
        query=query,
        k=DENSE_TOP_K
    )

    # Lexical retrieval
    bm25_retriever = initialize_bm25()

    bm25_results = bm25_retriever.invoke(query)

    # Combine rankings using RRF
    final_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=RRF_K,
        top_k=top_k
    )

    return final_results


# --------------------------------------------------
# 10. Test hybrid retrieval
# --------------------------------------------------

if __name__ == "__main__":

    query = "What happens when the fraud score exceeds 0.6?"

    results = hybrid_retrieval(
        query=query,
        top_k=FINAL_TOP_K
    )

    print(f"\nQuery: {query}")
    print(f"Retrieved chunks: {len(results)}")

    for index, (document, score) in enumerate(
        results,
        start=1
    ):

        print("\n" + "=" * 60)
        print(f"RESULT {index}")
        print("=" * 60)

        print(f"RRF Score: {score:.6f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Page: {document.metadata.get('page')}")

        print("\nContent:")
        print(document.page_content)