# rag_pipeline.py
import os
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


def create_rag_pipeline(docs):
    """
    Creates a Retrieval-Augmented Generation (RAG) pipeline using LangChain + HuggingFace.
    
    Args:
        docs (list): List of text documents to index
        
    Returns:
        RetrievalQA: The configured QA chain
    """
    
    # Ensure HuggingFace API token is set
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN environment variable not set. "
            "Get your token from https://huggingface.co/settings/tokens"
        )
    
    # Step 1: Create embeddings for documents
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # <-- ADD THIS LINE
    )
    
    # Step 2: Store them in FAISS vector database
    vectorstore = FAISS.from_texts(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Step 3: Load a Hugging Face LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )
    
    # Step 4: Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain