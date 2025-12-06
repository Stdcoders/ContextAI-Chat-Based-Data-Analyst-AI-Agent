import os
import pandas as pd
import json
import numpy as np
from typing import Optional, Dict, List
from collections import Counter
from langchain_core.tools import tool
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional langdetect import - fallback if not available
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    def detect(text):
        """Fallback language detection - assumes English"""
        return 'en'

# Optional LangChain imports - fallback if not available
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = False
    HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    FAISS = None

try:
    from langchain_community.document_loaders import (
        CSVLoader, UnstructuredExcelLoader, JSONLoader,
        TextLoader, UnstructuredPDFLoader, UnstructuredImageLoader, PyPDFLoader
    )
    LANGCHAIN_LOADERS_AVAILABLE = True
except ImportError:
    LANGCHAIN_LOADERS_AVAILABLE = False
    # Define fallbacks to avoid NameErrors
    CSVLoader, UnstructuredExcelLoader, JSONLoader = None, None, None
    TextLoader, UnstructuredPDFLoader, UnstructuredImageLoader, PyPDFLoader = None, None, None, None


# ================== FILE INGESTOR CLASS ==================
# This class contains the core, reusable logic for processing different file types.
# It is kept self-contained and does not interact with any global state.
class FileIngestor:
    def __init__(self, embedding_model=None, chunk_size=1000, chunk_overlap=200):
        self.datasets = {}
        self.vector_stores = {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if embedding_model:
            self.embedding_model = embedding_model
        elif HUGGINGFACE_EMBEDDINGS_AVAILABLE:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embedding_model = None

        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except Exception:
            self.text_splitter = None

    def from_csv(self, name: str, filepath: str, csv_args: Optional[Dict] = None) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        df = pd.read_csv(filepath, **(csv_args or {}))
        documents = [Document(page_content=df.to_csv(index=False), metadata={"source": filepath})]
        self._store_dataset(name, df, documents, "csv")
        return df

    def from_excel(self, name: str, filepath: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Excel file not found: {filepath}")
        
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        if isinstance(df, dict):
            df = list(df.values())[0]
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        
        documents = [Document(page_content=df.to_csv(index=False), metadata={"source": filepath})]
        self._store_dataset(name, df, documents, "excel")
        return df

    def from_json(self, name: str, filepath: str, jq_schema: str = '.') -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        df = None
        documents = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
            
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            
            documents = [Document(page_content=json.dumps(item), metadata={"source": filepath}) for item in (data if isinstance(data, list) else [data])]
        
        except json.JSONDecodeError:
            # Fallback for JSON Lines format (one JSON object per line)
            data_list = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data_list.append(json.loads(line))
            df = pd.DataFrame(data_list)
            documents = [Document(page_content=json.dumps(item), metadata={"source": filepath}) for item in data_list]

        if df is None:
             raise ValueError("Could not parse JSON file.")

        self._store_dataset(name, df, documents, "json")
        return df

    def from_text(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Text file not found: {filepath}")
        
        loader = TextLoader(filepath, encoding='utf-8')
        documents = loader.load()
        
        if self.text_splitter:
            split_documents = self.text_splitter.split_documents(documents)
        else:
            split_documents = documents
        
        if not split_documents:
            raise ValueError(f"No text extracted from {filepath}")
        
        df = pd.DataFrame({
            "content": [doc.page_content for doc in split_documents],
            "metadata": [json.dumps(doc.metadata, sort_keys=True, default=str) for doc in split_documents],
            "length": [len(doc.page_content) for doc in split_documents]
        })
        self._store_dataset(name, df, split_documents, "text")
        return df

    def from_pdf(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        if not LANGCHAIN_LOADERS_AVAILABLE:
            raise ImportError("PDF processing requires langchain-community. Please install it.")

        try:
            loader = UnstructuredPDFLoader(filepath)
            documents = loader.load()
        except Exception:
            loader = PyPDFLoader(filepath)
            documents = loader.load()

        if self.text_splitter:
            split_documents = self.text_splitter.split_documents(documents)
        else:
            split_documents = documents
            
        if not split_documents:
            raise ValueError(f"No text extracted from {filepath}")
            
        df = pd.DataFrame({
            "content": [doc.page_content for doc in split_documents],
            "metadata": [json.dumps(doc.metadata, sort_keys=True, default=str) for doc in split_documents],
            "page": [doc.metadata.get("page", 0) for doc in split_documents],
            "length": [len(doc.page_content) for doc in split_documents]
        })
        self._store_dataset(name, df, split_documents, "pdf")
        return df

    def from_image(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")
        if not LANGCHAIN_LOADERS_AVAILABLE:
            raise ImportError("Image processing requires langchain-community. Please install it.")
        
        loader = UnstructuredImageLoader(filepath)
        documents = loader.load()
        if not documents:
            raise ValueError(f"No text extracted from {filepath}")
            
        df = pd.DataFrame({
            "content": [doc.page_content for doc in documents],
            "metadata": [json.dumps(doc.metadata, sort_keys=True, default=str) for doc in documents],
            "length": [len(doc.page_content) for doc in documents]
        })
        self._store_dataset(name, df, documents, "image")
        return df

    def _store_dataset(self, name: str, df: pd.DataFrame, documents: List[Document], data_type: str):
        self.datasets[name] = {
            "data": df,
            "documents": documents,
            "profile": self._profile(df, documents, data_type),
        }
        if documents:
            self._create_vector_store(name, documents)

    def _create_vector_store(self, name: str, documents: List[Document]):
        if self.embedding_model and documents and FAISS_AVAILABLE:
            try:
                self.vector_stores[name] = FAISS.from_documents(documents, self.embedding_model)
            except Exception as e:
                print(f"Warning: Could not create vector store for {name}: {e}")

    def _profile(self, df: pd.DataFrame, documents: List[Document], data_type: str) -> Dict:
        profile = {
            "data_type": data_type,
            "n_rows": len(df) if df is not None else 0,
            "n_cols": df.shape[1] if hasattr(df, "shape") else 1,
            "columns": list(df.columns) if hasattr(df, "columns") else ["content"],
            "document_count": len(documents),
            "sample": df.head(3).to_dict(orient="records") if df is not None else []
        }

        if data_type in ["csv", "excel", "json"] and df is not None:
            try:
                profile["num_duplicates"] = int(df.duplicated().sum())
            except TypeError:
                safe_row_repr = df.apply(lambda r: json.dumps(r.to_dict(), sort_keys=True, default=str), axis=1)
                profile["num_duplicates"] = int(safe_row_repr.duplicated().sum())

            columns = []
            for col in df.columns:
                col_series = df[col]
                col_info = {
                    "name": col, "dtype": str(col_series.dtype),
                    "num_missing": int(col_series.isna().sum()),
                }
                columns.append(col_info)
            profile["columns"] = columns

        if data_type in ["text", "pdf"] and documents:
            text_lengths = [len(doc.page_content) for doc in documents if doc.page_content]
            languages = []
            if LANGDETECT_AVAILABLE:
                for doc in documents[:20]:
                    try:
                        if doc.page_content.strip():
                            languages.append(detect(doc.page_content[:200]))
                    except Exception:
                        languages.append('unknown')
            profile["detected_languages"] = dict(Counter(languages))
            
        return profile

    def get_profile(self, name: str) -> Dict:
        return self.datasets[name]["profile"]

    def get_dataset(self, name: str) -> pd.DataFrame:
        return self.datasets[name]["data"]


# ================== THE LANGCHAIN TOOL ==================
# This is the single, stateless function that your agent will call.
# It uses the FileIngestor class internally to do the heavy lifting.

@tool
def ingest_data(dataset_name: str, file_path: str, file_type: str) -> dict:
    """
    Ingests data from a file and returns a dataframe and a detailed data profile.
    Use this tool as the first step to load any new dataset into the system for analysis.
    
    Args:
        dataset_name (str): A unique name for the dataset, derived from the filename.
        file_path (str): The local path to the data file.
        file_type (str): The type of file. Supported types are 'csv', 'excel', 'json', 'pdf', 'text'.
    
    Returns:
        dict: A dictionary containing the pandas DataFrame under the key 'dataframe' and the data profile under the key 'profile'.
    """
    try:
        # 1. Instantiate the ingestor to perform the work.
        ingestor = FileIngestor()

        # 2. Call the appropriate method based on file_type
        df = getattr(ingestor, f"from_{file_type}")(dataset_name, file_path)
        
        profile = ingestor.get_profile(dataset_name)

        # 3. CRITICAL: Return a dictionary with the results. Do not modify any global state.
        print(f"✅ Tool 'ingest_data' successfully processed '{file_path}'.")
        return {"dataframe": df, "profile": profile}
        
    except Exception as e:
        # 4. Return errors in a structured way for the agent to handle.
        print(f"❌ Error in 'ingest_data' tool: {e}")
        return {"error": str(e)}