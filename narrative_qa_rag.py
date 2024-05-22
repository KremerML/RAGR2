import os
import pandas as pd
import openai
from typing import List, Dict, Any, Optional
from tqdm import tqdm  
import jsonlines

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

class DocumentLoader:
    def __init__(self, config):
        self.config = config

        self.qaps_path = config.get('qaps_path', None)
        self.summaries_path = config.get('summaries_path', None)
        self.model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.chunk_size = config.get('chunk_size', 1200)
        self.chunk_overlap = config.get('chunk_overlap', 0)
        self.vecdb_type = config.get('vecdb_type', 'FAISS')
        self.num_questions = config.get('num_questions', 500)
        self.min_len = config.get('min_len', 40)
        self.max_len = config.get('max_len', 100)

        self.output_qaps_file = os.path.join('/Users/kremerr/Documents/GitHub/RAGR2/archive/selected_qaps', f'long_ans_{self.num_questions}_qaps.csv')
        self.vecdb_path = os.path.join('/Users/kremerr/Documents/GitHub/RAGR2/vecdb_versions', f'{self.vecdb_type}_{self.num_questions}_{self.chunk_size}_long_ans')


    def load_data(self):
        self.qaps, self.doc_ids = self.question_selection()
        self.summaries = self.document_selection()

    def question_selection(self):
        if os.path.exists(self.output_qaps_file):
            logging.info("Using cached QAPs from %s", self.output_qaps_file)
            selected_qaps = pd.read_csv(self.output_qaps_file)
            doc_ids = set(selected_qaps['document_id'])
        else:
            logging.info("Selecting questions from %s", self.qaps_path)
            qaps = pd.read_csv(self.qaps_path)
            qaps = qaps.drop_duplicates(subset=['question'])
            qaps = qaps.dropna()
            # qaps['question_length'] = qaps['question'].apply(len)
            qaps['answer_length'] = qaps['answer1'].apply(len)
            qaps = qaps.query(f"{self.min_len} <= answer_length <= {self.max_len}")
            qaps = qaps.sample(frac=1).reset_index(drop=True)
            
            selected_qaps = qaps.head(self.num_questions)
            doc_ids = set(selected_qaps['document_id'])
            selected_qaps.to_csv(self.output_qaps_file, index=False)
            logging.info("%d questions selected", len(selected_qaps))

        self.output_summaries_file = os.path.join('/Users/kremerr/Documents/GitHub/RAGR2/archive/selected_summaries', f'{len(doc_ids)}_selected_summaries.csv')
        return selected_qaps, doc_ids
    
    def document_selection(self):
        if os.path.exists(self.output_summaries_file):
            logging.info("Using cached document summaries from %s", self.output_summaries_file)
            selected_summaries = pd.read_csv(self.output_summaries_file)
        else:
            logging.info("Selecting documents from %s", self.summaries_path)
            summaries = pd.read_csv(self.summaries_path)
            selected_summaries = summaries[summaries['document_id'].isin(self.doc_ids)]
            selected_summaries.pop('summary_tokenized')
            selected_summaries.pop('set')
            selected_summaries.to_csv(self.output_summaries_file, index=False)
            logging.info("%d documents selected", len(self.doc_ids))

        return selected_summaries

    def create_vecdb(self):
        logging.info("Initializing vector database at %s", self.vecdb_path)
        if os.path.exists(self.vecdb_path):
            logging.info("Vector store already exists, loading...")
            vecdb = FAISS.load_local(self.vecdb_path, HuggingFaceEmbeddings(model_name=self.model_name))
            return vecdb
        else:
            logging.info("Vector store not found, creating new...")
            if self.summaries.empty:
                logging.warning("Dataset is empty. Cannot create vector store.")
                return None
            
            embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            loader = DataFrameLoader(data_frame=self.summaries, page_content_column='summary')
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            vecdb = FAISS.from_documents(splits, embeddings)
            vecdb.save_local(self.vecdb_path)
            logging.info("%d documents added to vector store.", len(splits))
            return vecdb
    

def retrieve_evidence(query: str, vecdb, top_k: int = 2) -> List[Dict[str, Any]]:
    retriever = vecdb.as_retriever(search_kwargs={"k": top_k})
    retrieved_snippets = retriever.get_relevant_documents(query)
    evidences = [
        {"text": evidence.page_content, "metadata": evidence.metadata, "query": query}
        for evidence in retrieved_snippets
    ]
    logging.info(f'Retrieved evidence: {evidences}')
    return evidences
    


def csv_to_jsonl(csv_path: str, jsonl_path: str):
    # Load the data from CSV
    data = pd.read_csv(csv_path)
    
    # Open a jsonlines file to write the data
    with jsonlines.open(jsonl_path, mode='w') as writer:
        for _, row in data.iterrows():
            # Create the dictionary structure for each row
            entry = {
                "input_info": {
                    "claim": row['response'],
                    "question": row['question']
                }
            }
            # Write each dictionary as a separate line in the JSONL file
            writer.write(entry)
    print(f"Data has been written to {jsonl_path}")



# def create_vecdb_chroma() -> Optional[Chroma]:
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