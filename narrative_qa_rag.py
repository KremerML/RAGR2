import os
import pandas as pd
import openai
import uuid
from typing import List, Dict, Any, Optional
from tqdm import tqdm  

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.vectorstores import FAISS

openai.api_key = os.getenv("OPENAI_API_KEY")
DATASET_PATH = 'archive/narrative_qa/summaries.csv'
PERSIST_PATH = 'vecdb_versions/narrative_qa_faiss_vecdb'


def create_vecdb() -> Optional[FAISS]:
    """
    Initialize a FAISS vector store with documents from the dataset.
    If the vector store already exists, it will be loaded instead of creating a new one.
    Returns:
        FAISS: An initialized FAISS vector store instance, or None if the dataset is empty.
    """
    # Initialize embeddings and text splitter
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

    # Check if the vector store already exists
    if os.path.exists(PERSIST_PATH):
        print('FAISS vector store already exists, loading...')
        vecdb = FAISS.load_local(PERSIST_PATH, embeddings)
    else:
        print('FAISS vector store not found, creating new...')
        df = pd.read_csv(DATASET_PATH)
        if df.empty:
            print("Dataset is empty. Cannot create vector store.")
            return None
        df.pop('summary_tokenized')  # Drop the 'summary_tokenized' column if it exists
        loader = DataFrameLoader(data_frame=df, page_content_column='summary')
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        print("Documents loaded and split...")
        print("Adding texts to vector store...")
        vecdb = FAISS.from_documents(
            splits,
            embeddings
        )
        print("Documents added.")
        vecdb.save_local(PERSIST_PATH)
    return vecdb

# def create_vecdb() -> Optional[Chroma]:
#     """
#     Initialize a Chroma vector store with documents from the dataset.
#     If the vector store already exists, it will be loaded instead of creating a new one.

#     Returns:
#         Chroma: An initialized Chroma vector store instance, or None if the dataset is empty.
#     """
#     # Initialize embeddings and text splitter
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

#     # Check if the vector store already exists
#     if os.path.exists(PERSIST_PATH):
#         print('Chromadb already exists, loading...')
#         vecdb = Chroma(persist_directory=PERSIST_PATH, 
#                        embedding_function=embeddings,
#                        collection_name='narrative_qa_collection',
#                        client_settings = Settings(allow_reset=True))
#     else:
#         print('Chromadb not found, creating new...')
#         vecdb = Chroma(
#             persist_directory=PERSIST_PATH,
#             embedding_function=embeddings,
#             collection_name='narrative_qa_collection',
#             client_settings = Settings(allow_reset=True)
#         )

#         # Load summaries.csv
#         df = pd.read_csv(DATASET_PATH)
#         if df.empty:
#             print("Dataset is empty. Cannot create vector store.")
#             return None

#         df.pop('summary_tokenized')  # Drop the 'summary_tokenized' column if it exists
#         loader = DataFrameLoader(data_frame=df, page_content_column='summary')
#         docs = loader.load()

#         # Split documents
#         splits = text_splitter.split_documents(docs)
#         print("Documents loaded and split...")

#         # Add documents to the vector store
#         print("Adding texts to vector store...")
#         vecdb.add_documents(splits)
#         print("Documents added.")
#         vecdb.persist()

#     return vecdb


def retrieve_evidence(
    query: str,
    vecdb: Chroma,
    top_k: int = 2
) -> List[Dict[str, Any]]:
    """Retrieve the most relevant evidence from the vector store using LangChain.

    Args:
        query: The query string for which to retrieve evidence.
        vecdb: The initialized Chroma vector store.
        top_k: The number of top results to retrieve.

    Returns:
        evidences: A list of evidences, each evidence is a dictionary containing 'page_content' and 'metadata'.
    """
    retriever = vecdb.as_retriever(search_kwargs={"k": top_k})
    evidences = retriever.get_relevant_documents(query)
    evidences = [
        {
            "text": evidence.page_content,
            "metadata": evidence.metadata,
            "query": query
        }
        for evidence in evidences
    ]
    return evidences
