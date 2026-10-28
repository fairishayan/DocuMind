import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def process_pdf(pdf_path):
    print(f"1. Loading PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("2. Splitting text into chunks...")
    # Text ko chhote chunks me todna (taaki AI easily padh sake)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Created {len(chunks)} chunks.")
    
    print("3. Creating Embeddings (this may take a minute)...")
    # Free local model use kar rahe hain text ko numbers me badalne ke liye
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("4. Saving to ChromaDB...")
    # Data ko vector database me save karna
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db" # Data yahan save hoga
    )
    print("✅ Success! Data is ingested and saved in Vector DB.")

if __name__ == "__main__":
    # Make sure you have a sample.pdf in this folder
    if os.path.exists("sample.pdf"):
        process_pdf("sample.pdf")
    else:
        print("❌ Error: Please put a file named 'sample.pdf' in this folder.")
