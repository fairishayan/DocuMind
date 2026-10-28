from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def search_pdf(query):
    print(f"\n🔍 Searching for: '{query}'\n")
    
    # 1. Load the same embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to the existing ChromaDB
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 3. Search for top 2 most relevant chunks
    results = vectorstore.similarity_search(query, k=2)
    
    if not results:
        print("No relevant context found.")
        return
        
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} (Page {doc.metadata.get('page', 'Unknown')}) ---")
        # Sirf pehle 300 characters print kar rahe hain taaki screen clean rahe
        print(doc.page_content[:300] + "...\n")

if __name__ == "__main__":
    # Test query
    search_pdf("What is this document about?")
