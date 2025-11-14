import os
import tempfile
import hashlib
from typing import List, Any
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import qdrant_client


def load_uploaded_docs(uploaded_files: List[Any]):
    docs = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            if os.path.getsize(temp_file_path) == 0:
                st.error(f"Uploaded file {uploaded_file.name} is empty and was skipped.")
                os.unlink(temp_file_path)
                continue

            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".txt":
                loader = TextLoader(temp_file_path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                os.unlink(temp_file_path)
                continue

            file_docs = loader.load()
            for doc in file_docs:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["source"] = uploaded_file.name
            docs.extend(file_docs)
            os.unlink(temp_file_path)
        except Exception as e:
            st.error(f"Failed to load uploaded file {uploaded_file.name}: {str(e)}")
    return docs


def split_documents(docs: List[Any], chunk_size: int = 250):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=100
    )
    return text_splitter.split_documents(docs)


def build_qdrant_vectorstore(doc_splits: List[Any], google_api_key: str, qdrant_url: str, qdrant_api: str, collection_name: str = "agentic_collection"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    client = QdrantClient(
        qdrant_url,
        api_key=qdrant_api
    )

    collection_config = qdrant_client.http.models.VectorParams(
        size=768,
        distance=qdrant_client.http.models.Distance.COSINE
    )

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=collection_config
    )

    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    texts = [doc.page_content for doc in doc_splits]
    vectorstore.add_texts(texts)
    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_knowledge",
        "Search and return information from the provided knowledge sources.",
    )

    return retriever, retriever_tool


def calculate_knowledge_hash(files):
    content = ""
    for file in files:
        content += file.getvalue().decode(errors="ignore")
    return hashlib.md5(content.encode()).hexdigest()