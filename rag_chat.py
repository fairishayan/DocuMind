import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

def load_rag_system():
    print("1. Loading Embeddings & Vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    print("2. Loading Local LLM (TinyLlama)...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        max_new_tokens=150, 
        temperature=0.1,    
        device=-1           
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("3. Setting up Prompt Template...")
    template = """<|system|>
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say "I don't know", don't make up an answer.</s>
<|user|>
Context: {context}
Question: {question}</s>
<|assistant|>"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    print("4. Creating the RAG Chain (LCEL)...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Modern LCEL Pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_question(qa_chain, query):
    print(f"\n🤔 Question: {query}")
    print("⏳ Thinking (CPU inference, please wait)...")
    
    response = qa_chain.invoke(query)
    
    print("\n🤖 AI Answer:")
    # TinyLlama sometimes outputs the whole prompt, so we clean it
    clean_response = response.split("<|assistant|>")[-1].strip()
    print(clean_response)
    print("-" * 50)

if __name__ == "__main__":
    rag_system = load_rag_system()
    ask_question(rag_system, "What is this document about?")
