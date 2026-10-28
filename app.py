import streamlit as st
import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

st.set_page_config(page_title="DocuMind", page_icon="📄")
st.title("📄 DocuMind: Chat with your PDF")

@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # UPGRADED MODEL: Qwen 2.5 (1.5B)
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        tokenizer=model_id, 
        max_new_tokens=250, 
        temperature=0.1, 
        device=-1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return embeddings, llm

embeddings, llm = load_models()

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
FAISS_PATH = "faiss_index"

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Delete old DB to prevent mixing data
            if os.path.exists(FAISS_PATH):
                shutil.rmtree(FAISS_PATH)
                
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            # Use FAISS instead of Chroma
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(FAISS_PATH)
            st.success("PDF Processed! Ask your questions below.")

    if os.path.exists(FAISS_PATH):
        # Load FAISS
        vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        template = """<|im_start|>system
You are a strict assistant. Answer the user's question using ONLY the provided context. 
Do not make up information. If the answer is not in the context, say "I don't know based on the document."<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}<|im_end|>
<|im_start|>assistant
"""
        
        prompt = PromptTemplate.from_template(template)
        def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(user_query)
                    clean_response = response.split("<|im_start|>assistant\n")[-1].strip()
                    st.markdown(clean_response)
                    st.session_state.messages.append({"role": "assistant", "content": clean_response})
