# DocuMind: A Privacy-Focused RAG-Based PDF Chatbot
DocuMind is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents in natural language. Instead of manually searching through hundreds of pages, users can interact with their documents and receive relevant, context-based answers.
The project combines semantic search with a locally running Large Language Model (LLM) to make document analysis more accessible while keeping document processing and inference on the user's machine. It is designed to work with large PDFs, making it useful for studying textbooks, reviewing technical documentation, researching academic papers, and extracting information from lengthy reports.

## Key Features
- **PDF-Based Question Answering:** Upload PDF documents and ask questions directly about their contents.
- **Semantic Search:** Retrieve relevant passages based on meaning rather than relying only on exact keyword matches.
- **Local AI Inference:** Run Qwen 2.5 (1.5B Instruct) locally without requiring an external LLM API.
- **Efficient Vector Search:** Use FAISS to store and retrieve document embeddings efficiently.
- **Context-Grounded Responses:** Generate answers using retrieved document passages to reduce unsupported responses.
- **Interactive Interface:** Use Streamlit to interact with documents through a simple chat interface.
- **Local Document Processing:** Keep document processing, embeddings, vector storage, and LLM inference on the local machine, subject to the application's implementation and configuration.

## How It Works

**1. Document Ingestion and Chunking**
When a user uploads a PDF, the application extracts its text and divides it into smaller, manageable chunks. Chunking allows the retrieval system to search specific sections instead of processing the entire document for every question.

**2. Text Embedding Generation**
Each text chunk is converted into a numerical vector using Hugging Face's `all-MiniLM-L6-v2` embedding model. These vectors represent semantic information, helping the system identify passages relevant to a user's question.

**3. Local Vector Storage with FAISS**
The generated embeddings are indexed using FAISS, enabling efficient similarity searches across document chunks. When a question is submitted, the system searches the index to retrieve the passages most relevant to the query.

**4. Context-Aware Answer Generation**
The retrieved passages are combined with the user's question and passed to the locally running Qwen 2.5 (1.5B Instruct) model. The model uses this context to generate a response grounded in the available document content rather than relying solely on its pretrained knowledge.

**5. Interactive Question Answering**
The Streamlit frontend presents the conversation in an accessible interface, allowing users to ask follow-up questions and explore the information contained in their PDFs.

## Technology Stack
- **Programming Language:** Python
- **Application Framework:** Streamlit
- **LLM Integration:** LangChain Expression Language (LCEL)
- **Language Model:** Qwen 2.5 (1.5B Instruct)
- **Embedding Model:** Hugging Face `all-MiniLM-L6-v2`
- **Vector Database:** FAISS
- **Core Approach:** Retrieval-Augmented Generation (RAG)

## Installation and Setup
Clone the repository, install the required dependencies, and launch the application locally.

```
git clone <repository-url>
cd DocuMind
pip install -r requirements.txt
streamlit run app.py
```
Once the application starts, open the local URL displayed in the terminal to access the interface and begin interacting with PDF documents.

## Privacy and Performance Considerations
DocuMind is designed around local inference, eliminating the need to send document content to an external LLM API. Its performance depends on the available CPU, RAM, document size, and the number of text chunks processed during retrieval.
Because the language model runs on the CPU, response generation may take longer on machines with limited resources. The quality of the answers also depends on PDF text extraction, chunking strategy, retrieval relevance, and the model's ability to interpret the supplied context.
Local processing helps improve privacy, but it does not automatically guarantee complete data security. Actual privacy depends on the application's implementation, logging, temporary-file handling, and any external services or telemetry it uses.

## Project Objective
The goal of DocuMind is to demonstrate how RAG can connect document retrieval with local language-model inference to create a practical document question-answering system. By combining semantic embeddings, vector similarity search, and context-aware generation, the project provides a foundation for building AI-powered tools that help users find and understand information in large documents without depending on external LLM APIs.
