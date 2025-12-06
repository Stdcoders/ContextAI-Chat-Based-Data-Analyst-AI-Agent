import os
import pandas as pd
import json
import numpy as np
from typing import Optional, Dict, List
from collections import Counter

# Optional langdetect import - fallback if not available
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    def detect(text):
        """Fallback language detection - assumes English"""
        return 'en'

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        TextLoader, UnstructuredPDFLoader, UnstructuredImageLoader
    )
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_LOADERS_AVAILABLE = True
except ImportError:
    LANGCHAIN_LOADERS_AVAILABLE = False
    # Fallback classes - will handle files without LangChain
    CSVLoader = None
    UnstructuredExcelLoader = None
    JSONLoader = None
    TextLoader = None
    UnstructuredPDFLoader = None
    UnstructuredImageLoader = None
    PyPDFLoader = None
from tabulate import tabulate
from utils.state import STATE

# ================== FILE INGESTOR ==================
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
            self.embedding_model = None  # Vector search will be disabled

        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except ImportError:
            # Fallback text splitter if LangChain not available
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
        
        # Read Excel file
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        # Handle case where multiple sheets are returned as dict
        if isinstance(df, dict):
            if len(df) == 1:
                # Single sheet in dict format - get the DataFrame
                df = list(df.values())[0]
            else:
                # Multiple sheets - combine them or pick the first one
                print(f"Multiple sheets found: {list(df.keys())}. Using the first sheet.")
                df = list(df.values())[0]
        
        # Ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        
        # Create document
        documents = [Document(page_content=df.to_csv(index=False), metadata={"source": filepath})]
        self._store_dataset(name, df, documents, "excel")
        return df

    def from_json(self, name: str, filepath: str, jq_schema: str = '.') -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        if LANGCHAIN_LOADERS_AVAILABLE and JSONLoader:
            try:
                loader = JSONLoader(file_path=filepath, jq_schema=jq_schema, text_content=False)
                documents = loader.load()
                try:
                    data = [json.loads(doc.page_content) for doc in documents]
                    df = pd.DataFrame(data)
                except Exception:
                    df = pd.DataFrame({
                        "content": [doc.page_content for doc in documents],
                        "metadata": [json.dumps(doc.metadata, sort_keys=True, default=str) for doc in documents]
                    })
            except Exception as e:
                print(f"⚠️ LangChain JSONLoader failed: {e}. Using fallback method.")
                # Fall through to manual processing
                documents = None
                df = None
        else:
            documents = None
            df = None
        
        # Fallback or manual processing when LangChain fails
        if df is None:
            # Try to detect JSON format first
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)  # Reset file pointer
                
                try:
                    # Try to parse first line as JSON to detect format
                    json.loads(first_line)
                    is_jsonlines = True
                    print("📝 Detected JSON Lines format")
                except json.JSONDecodeError:
                    is_jsonlines = False
                    print("📝 Detected standard JSON format")
                
                if is_jsonlines:
                    # Handle JSON Lines format (one JSON object per line)
                    data = []
                    line_number = 0
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                line_number += 1
                                obj = json.loads(line)
                                data.append(obj)
                            except json.JSONDecodeError as e:
                                print(f"⚠️ Skipping invalid JSON on line {line_number}: {e}")
                                continue
                    
                    if not data:
                        raise ValueError("No valid JSON objects found in JSON Lines file")
                    
                    df = pd.DataFrame(data)
                    print(f"📝 Successfully loaded {len(data)} records from JSON Lines format")
                    
                else:
                    # Handle standard JSON format
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON format: {e}")
                    
                    # Handle different JSON structures
                    if isinstance(data, list):
                        # JSON is already a list of records
                        df = pd.DataFrame(data)
                        print(f"📝 Loaded {len(data)} records from JSON array")
                    elif isinstance(data, dict):
                        # Check if it's a nested structure with a 'data' key
                        if 'data' in data and isinstance(data['data'], list):
                            # Extract the data array for DataFrame creation
                            df = pd.DataFrame(data['data'])
                            print(f"📝 Extracted {len(data['data'])} records from nested JSON structure")
                            if 'dataset_name' in data:
                                print(f"📊 Dataset: {data['dataset_name']}")
                            if 'description' in data:
                                print(f"📋 Description: {data['description']}")
                        else:
                            # JSON is a single object - convert to single-row DataFrame
                            df = pd.DataFrame([data])
                            print(f"📝 Created DataFrame from single JSON object")
                    else:
                        # Fallback for other types (numbers, strings, etc.)
                        df = pd.DataFrame([{'content': str(data)}])
                        print(f"📝 Converted {type(data).__name__} to DataFrame")
                
                # Create simple documents without LangChain
                if isinstance(data, list) and len(data) > 0:
                    # For arrays, create documents from individual items
                    documents = [Document(page_content=json.dumps(item), metadata={"source": filepath, "index": i}) 
                               for i, item in enumerate(data[:100])]  # Limit to first 100 for memory
                else:
                    # For single objects or other formats
                    documents = [Document(page_content=json.dumps(data) if 'data' in locals() else first_line, 
                                        metadata={"source": filepath})]
        
        self._store_dataset(name, df, documents, "json")
        return df
  

    def from_text(self, name: str, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Text file not found: {filepath}")
        
        if LANGCHAIN_LOADERS_AVAILABLE and TextLoader:
            # Try UTF-8 encoding first, then fallback to default
            try:
                loader = TextLoader(filepath, encoding='utf-8')
                documents = loader.load()
            except UnicodeDecodeError:
                try:
                    loader = TextLoader(filepath, encoding='cp1252')
                    documents = loader.load()
                except UnicodeDecodeError:
                    # Final fallback to error handling
                    loader = TextLoader(filepath, encoding='utf-8', errors='ignore')
                    documents = loader.load()
            if self.text_splitter:
                split_documents = self.text_splitter.split_documents(documents)
            else:
                split_documents = documents
        else:
            # Fallback: direct text loading without LangChain
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple text splitting if no LangChain
            chunk_size = 1000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            split_documents = [Document(page_content=chunk, metadata={"source": filepath}) for chunk in chunks if chunk.strip()]
        
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
            raise ImportError("PDF processing requires langchain-community. Please install it: pip install langchain-community")
        
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
            raise ImportError("Image processing requires langchain-community. Please install it: pip install langchain-community")
        
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

    # ---------------- private helpers ----------------
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
                # Continue without vector store

    def _profile(self, df: pd.DataFrame, documents: List[Document], data_type: str) -> Dict:
        """Generate a comprehensive profile for the dataset (structured + unstructured)."""
        profile = {
            "data_type": data_type,
            "n_rows": len(df) if df is not None else 0,
            "n_cols": df.shape[1] if hasattr(df, "shape") else 1,
            "columns": list(df.columns) if hasattr(df, "columns") else ["content"],
            "document_count": len(documents),
            "sample": df.head(3).to_dict(orient="records") if df is not None else []
        }

        # Structured data profiling
        if data_type in ["csv", "excel", "json"] and df is not None:
            # Safely compute duplicate rows even when cells contain unhashable types (e.g., lists)
            try:
                profile["num_duplicates"] = int(df.duplicated().sum())
            except TypeError:
                safe_row_repr = df.apply(lambda r: json.dumps(r.to_dict(), sort_keys=True, default=str), axis=1)
                profile["num_duplicates"] = int(safe_row_repr.duplicated().sum())

            def safe_nunique(series: pd.Series) -> int:
                try:
                    return int(series.nunique(dropna=True))
                except TypeError:
                    return int(series.astype(str).nunique(dropna=True))

            def safe_top_values(series: pd.Series) -> Dict:
                try:
                    return series.value_counts(dropna=True).head(5).to_dict()
                except TypeError:
                    return series.astype(str).value_counts(dropna=True).head(5).to_dict()

            columns = []
            for col in df.columns:
                col_series = df[col]
                col_info = {
                    "name": col,
                    "dtype": str(col_series.dtype),
                    "num_missing": int(col_series.isna().sum()),
                    "pct_missing": float(col_series.isna().mean()),
                    "num_unique": safe_nunique(col_series),
                }
                if pd.api.types.is_numeric_dtype(col_series):
                    col_info.update({
                        "min": col_series.min(),
                        "max": col_series.max(),
                        "mean": col_series.mean(),
                        "median": col_series.median(),
                        "std": col_series.std(),
                    })
                    std = col_series.std()
                    if std and std != 0:
                        z_scores = np.abs((col_series - col_series.mean()) / std)
                        col_info["num_outliers"] = int((z_scores > 3).sum())
                    else:
                        col_info["num_outliers"] = 0
                elif pd.api.types.is_object_dtype(col_series):
                    col_info["top_values"] = safe_top_values(col_series)
                elif np.issubdtype(col_series.dtype, np.datetime64):
                    col_info.update({
                        "min_date": str(col_series.min()),
                        "max_date": str(col_series.max()),
                    })
                columns.append(col_info)
            profile["columns"] = columns

        # Text / PDF profiling
 
        if data_type in ["text", "pdf"] and documents:
            text_lengths = [len(doc.page_content) for doc in documents if doc.page_content]
            token_counts = [len(doc.page_content.split()) for doc in documents if doc.page_content]
            languages = []
            if LANGDETECT_AVAILABLE:
                for doc in documents[:20]:
                    try:
                        if doc.page_content.strip():  # Only detect if content exists
                            languages.append(detect(doc.page_content[:200]))
                    except Exception:
                        languages.append('unknown')  # Fallback for detection failures
            else:
                languages = ['en'] * min(len(documents), 20)  # Default to English when langdetect unavailable
            profile.update({
                "avg_text_length": float(np.mean(text_lengths)) if text_lengths else 0,
                "min_text_length": int(min(text_lengths)) if text_lengths else 0,
                "max_text_length": int(max(text_lengths)) if text_lengths else 0,
                "avg_token_count": float(np.mean(token_counts)) if token_counts else 0,
                "detected_languages": dict(Counter(languages)),
            })


        # Image profiling
        if data_type == "image" and documents:
            image_meta = [doc.metadata for doc in documents if hasattr(doc, "metadata")]
            widths = [meta.get("width", 0) for meta in image_meta if "width" in meta]
            heights = [meta.get("height", 0) for meta in image_meta if "height" in meta]
            sizes = [meta.get("size_kb", 0) for meta in image_meta if "size_kb" in meta]
            profile.update({
                "num_images": len(documents),
                "avg_width": float(np.mean(widths)) if widths else None,
                "avg_height": float(np.mean(heights)) if heights else None,
                "avg_file_size_kb": float(np.mean(sizes)) if sizes else None,
            })

        return profile

    def get_profile(self, name: str) -> Dict:
        if name not in self.datasets:
            raise KeyError(f"No dataset named '{name}'")
        return self.datasets[name]["profile"]

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())

    def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.datasets:
            raise KeyError(f"No dataset named '{name}'")
        return self.datasets[name]["data"]


# ================== NODE (stateless wrapper) ==================
def data_ingestion_node(dataset_name: str, file_path: str, file_type: str,
                        csv_args: Optional[dict] = None, sheet_name: Optional[str] = None):
    """
    Stateful ingestion node that stores results in global STATE.
    """
    ingestor = FileIngestor()

    if file_type == "csv":
        df = ingestor.from_csv(dataset_name, file_path, csv_args)
    elif file_type == "excel":
        df = ingestor.from_excel(dataset_name, file_path, sheet_name)
    elif file_type == "json":
        df = ingestor.from_json(dataset_name, file_path)
    elif file_type == "text":
        df = ingestor.from_text(dataset_name, file_path)
    elif file_type == "pdf":
        df = ingestor.from_pdf(dataset_name, file_path)
    elif file_type == "image":
        df = ingestor.from_image(dataset_name, file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    profile = ingestor.get_profile(dataset_name)

    # STORE IN GLOBAL STATE
    STATE.datasets[dataset_name] = df
    STATE.profiles[dataset_name] = profile
    
    # Also store the vector store if you need it later
    if hasattr(ingestor, 'vector_stores') and dataset_name in ingestor.vector_stores:
        # You might want to store vector stores in STATE too
        pass

    print(f"\n✅ Dataset '{dataset_name}' ingested successfully with {len(df)} records.")
    print("\n📊 Dataset Profile Summary:")

    if "columns" in profile and isinstance(profile["columns"], list) and all(isinstance(c, dict) for c in profile["columns"]):
        df_profile = pd.DataFrame(profile["columns"])
        print(tabulate(df_profile, headers="keys", tablefmt="pretty", showindex=False))
    else:
        print(json.dumps(profile, indent=2, default=str))

    return df, profile