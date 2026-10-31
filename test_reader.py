from src.rag_langchain_chatbot.fileReader import read_multiple_pdfs

documents = read_multiple_pdfs("data/pdfs")

print(f"\nTotal pages loaded: {len(documents)}")