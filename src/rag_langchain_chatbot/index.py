
import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_langchain_chatbot.fileReader import read_multiple_pdfs


# Configuration
PDF_DIRECTORY = "data/pdfs"
CHROMA_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "claimeye_documents"


# 1. Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# 2. Initialize persistent vector database
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIRECTORY
)


# 3. Load PDF documents
documents = read_multiple_pdfs(PDF_DIRECTORY)

print(f"Pages loaded: {len(documents)}")


# 4. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks generated: {len(chunks)}")


# 5. Generate deterministic IDs
def generate_chunk_id(document, index):
    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")

    content = (
        f"{source}|{page}|{index}|{document.page_content}"
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


chunk_ids = [
    generate_chunk_id(chunk, index)
    for index, chunk in enumerate(chunks)
]


# 6. Insert only new chunks
existing_ids = set(vector_store.get(ids=chunk_ids)["ids"])

new_chunks = []
new_ids = []

for chunk, chunk_id in zip(chunks, chunk_ids):
    if chunk_id not in existing_ids:
        new_chunks.append(chunk)
        new_ids.append(chunk_id)

if new_chunks:
    vector_store.add_documents(
        documents=new_chunks,
        ids=new_ids
    )

    print(f"{len(new_chunks)} new chunks indexed.")

else:
    print("All chunks are already indexed.")


print(
    f"Total chunks in ChromaDB: "
    f"{vector_store._collection.count()}"
)
