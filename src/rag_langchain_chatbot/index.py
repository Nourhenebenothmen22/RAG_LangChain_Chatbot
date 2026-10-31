
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_langchain_chatbot.fileReader import read_multiple_pdfs


# 1. Initialize the Sentence Transformers embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# 2. Initialize the vector store
vector_store = InMemoryVectorStore(
    embedding=embeddings
)


# 3. Load PDF documents
documents = read_multiple_pdfs("data/pdfs")

print(f"Number of pages loaded: {len(documents)}")


# 4. Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks generated: {len(chunks)}")


# 5. Generate embeddings and store documents
vector_store.add_documents(documents=chunks)

print("Documents successfully embedded and stored!")


# 6. Test semantic search
results = vector_store.similarity_search(
    query="What is the main objective of the ClaimEye AI project?",
    k=3
)

print("\nSearch Results:")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
