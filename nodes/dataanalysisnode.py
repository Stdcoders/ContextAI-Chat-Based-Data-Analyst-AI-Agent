import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
from typing import Dict, Any, List

from utils.state import STATE  # ✅ import global workflow state
from gemma_llm import GemmaLLM  # ✅ import Gemma LLM


def data_cleaning_analysis_node(df: pd.DataFrame, dataset_name: str):
    """
    LLM-driven intelligent data cleaning and analysis node.
    Uses Gemma LLM to determine optimal cleaning strategies.
    """

    print(f"🧹 Starting intelligent cleaning for dataset: {dataset_name}")
    
    # Initialize Gemma LLM for cleaning decisions
    llm = GemmaLLM(temperature=0.2, max_tokens=600)
    
    cleaning_report = {
        'original_shape': df.shape,
        'cleaning_method': 'llm_driven',
        'steps_performed': [],
        'llm_recommendations': {},
        'issues_found': [],
        'improvements_made': []
    }

    # Detect data type and structure
    is_text_data = "content" in df.columns and (df.shape[1] <= 5 or "metadata" in df.columns)
    
    if is_text_data:
        print("📄 Detected unstructured text data (PDF/Text)")
        cleaned_df, text_analysis = llm_clean_text_data(df, dataset_name, llm, cleaning_report)
        analysis = text_analysis
    else:
        print("📊 Detected structured data (CSV/Excel/JSON)")
        cleaned_df, structured_analysis = llm_clean_structured_data(df, dataset_name, llm, cleaning_report)
        analysis = structured_analysis

    # Update cleaning report
    cleaning_report['final_shape'] = cleaned_df.shape
    cleaning_report['cleaning_success'] = True

    # ✅ Update STATE - preserve original profile and add cleaning info
    print(f"🔍 Debug - Storing cleaned dataset '{dataset_name}' in STATE.datasets")
    STATE.datasets[dataset_name] = cleaned_df
    STATE.analysis[dataset_name] = analysis
    print(f"🔍 Debug - STATE.datasets now contains: {list(STATE.datasets.keys())}")
    
    # Preserve original profile and add cleaning info
    if dataset_name in STATE.profiles:
        # Update existing profile with cleaning information
        STATE.profiles[dataset_name].update({
            'cleaning_report': cleaning_report,
            'cleaned': True,
            'final_shape': cleaned_df.shape
        })
        print(f"📋 Updated existing profile for '{dataset_name}'")
    else:
        # This should not happen, but if it does, create a basic profile with cleaning info
        print(f"⚠️ Profile missing for '{dataset_name}' - creating basic profile")
        STATE.profiles[dataset_name] = {
            'data_type': 'unknown',
            'n_rows': cleaned_df.shape[0],
            'n_cols': cleaned_df.shape[1],
            'columns': cleaned_df.columns.tolist(),
            'cleaning_report': cleaning_report,
            'cleaned': True
        }

    print(f"✅ Intelligent cleaning completed for '{dataset_name}'. Final shape: {cleaned_df.shape}")
    print(f"📊 Cleaning improvements: {len(cleaning_report['improvements_made'])} enhancements made")
    
    # Save state after cleaning to ensure persistence
    try:
        import pickle
        import os
        STATE_FILE = "memory/state.pkl"
        os.makedirs("memory", exist_ok=True)
        with open(STATE_FILE, "wb") as f:
            pickle.dump(STATE, f)
        print(f"💾 Debug - Saved STATE after cleaning (datasets: {list(STATE.datasets.keys())})")
    except Exception as e:
        print(f"⚠️ Warning - Could not save state after cleaning: {e}")
    
    return STATE


# ========== LLM-DRIVEN CLEANING FUNCTIONS ==========

def llm_clean_text_data(df: pd.DataFrame, dataset_name: str, llm: GemmaLLM, cleaning_report: Dict[str, Any]):
    """LLM-driven cleaning for unstructured text data (PDFs, text files)."""
    print("🤖 Using LLM to analyze and clean text data...")
    
    # Analyze data structure and content
    data_profile = {
        'columns': list(df.columns),
        'shape': df.shape,
        'sample_content': df['content'].iloc[:3].tolist() if 'content' in df.columns else [],
        'has_metadata': 'metadata' in df.columns,
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Ask LLM for cleaning recommendations
    cleaning_prompt = f"""You are a data cleaning expert for text/document data. Analyze this dataset and recommend cleaning steps:

DATA PROFILE:
- Shape: {data_profile['shape']} (rows, columns)
- Columns: {data_profile['columns']}
- Has metadata: {data_profile['has_metadata']}
- Sample content: {str(data_profile['sample_content'])[:500]}...

Provide specific cleaning recommendations for text data:
1. Should we clean/normalize the text content?
2. How to handle metadata columns?
3. What text processing steps are needed?
4. Any columns to remove or transform?

Respond with actionable steps."""
    
    cleaning_advice = llm(cleaning_prompt)
    cleaning_report['llm_recommendations']['text_cleaning'] = cleaning_advice
    cleaning_report['steps_performed'].append("LLM analysis of text data")
    
    print(f"📝 LLM Cleaning Advice: {cleaning_advice[:200]}...")
    
    # Apply intelligent text cleaning
    cleaned_df = df.copy()
    
    if 'content' in cleaned_df.columns:
        print("🧹 Applying intelligent text cleaning...")
        
        # Clean text content based on LLM recommendations
        if "clean" in cleaning_advice.lower() or "normalize" in cleaning_advice.lower():
            cleaned_df['cleaned_content'] = cleaned_df['content'].apply(lambda x: intelligent_text_clean(x, llm))
            cleaning_report['improvements_made'].append("Applied LLM-guided text normalization")
        else:
            cleaned_df['cleaned_content'] = cleaned_df['content'].apply(basic_text_clean)
            cleaning_report['improvements_made'].append("Applied basic text cleaning")
        
        # Handle metadata intelligently
        if 'metadata' in cleaned_df.columns:
            if "remove" in cleaning_advice.lower() and "metadata" in cleaning_advice.lower():
                print("🗑️ Removing metadata column as recommended by LLM")
                cleaned_df = cleaned_df.drop('metadata', axis=1)
                cleaning_report['improvements_made'].append("Removed metadata column per LLM recommendation")
            else:
                cleaning_report['improvements_made'].append("Preserved metadata column")
    
    # Generate text analysis
    if 'cleaned_content' in cleaned_df.columns:
        word_counts = cleaned_df['cleaned_content'].str.split().apply(len)
        analysis = {
            'document_count': len(cleaned_df),
            'avg_doc_length': word_counts.mean(),
            'min_doc_length': word_counts.min(),
            'max_doc_length': word_counts.max(),
            'total_words': word_counts.sum(),
            'content_quality': 'high' if word_counts.mean() > 50 else 'medium'
        }
    else:
        analysis = {'document_count': len(cleaned_df), 'content_quality': 'basic'}
    
    return cleaned_df, analysis

def llm_clean_structured_data(df: pd.DataFrame, dataset_name: str, llm: GemmaLLM, cleaning_report: Dict[str, Any]):
    """LLM-driven cleaning for structured data (CSV, Excel, JSON)."""
    print("🤖 Using LLM to analyze and clean structured data...")
    
    # Create comprehensive data profile
    data_profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'data_types': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'sample_data': df.head(3).to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'text_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Ask LLM for comprehensive cleaning strategy
    cleaning_prompt = f"""You are an expert data analyst. Analyze this structured dataset and create a cleaning plan:

DATA PROFILE:
- Shape: {data_profile['shape']}
- Columns: {data_profile['columns']}
- Data types: {data_profile['data_types']}
- Missing values: {data_profile['missing_values']}
- Duplicates: {data_profile['duplicates']}
- Numeric columns: {data_profile['numeric_columns']}
- Text columns: {data_profile['text_columns']}
- Sample: {str(data_profile['sample_data'])[:500]}...

Provide a structured cleaning plan:
1. How to handle missing values for each column type?
2. Should duplicates be removed?
3. Any data type conversions needed?
4. How to handle outliers in numeric columns?
5. Text column cleaning requirements?
6. New features to create?

Be specific and actionable."""
    
    cleaning_strategy = llm(cleaning_prompt)
    cleaning_report['llm_recommendations']['structured_cleaning'] = cleaning_strategy
    cleaning_report['steps_performed'].append("LLM analysis of structured data")
    
    print(f"📈 LLM Cleaning Strategy: {cleaning_strategy[:200]}...")
    
    # Apply LLM-recommended cleaning steps
    cleaned_df = df.copy()
    
    # 1. Handle missing values intelligently
    if data_profile['missing_values'] and any(v > 0 for v in data_profile['missing_values'].values()):
        print("🔄 Applying intelligent missing value handling...")
        cleaned_df = llm_handle_missing_values(cleaned_df, cleaning_strategy, cleaning_report)
    
    # 2. Remove duplicates if recommended
    if data_profile['duplicates'] > 0 and "remove" in cleaning_strategy.lower() and "duplicate" in cleaning_strategy.lower():
        print(f"🗑️ Removing {data_profile['duplicates']} duplicates as recommended")
        cleaned_df = cleaned_df.drop_duplicates()
        cleaning_report['improvements_made'].append(f"Removed {data_profile['duplicates']} duplicate rows")
    
    # 3. Handle data type conversions
    if "convert" in cleaning_strategy.lower() or "type" in cleaning_strategy.lower():
        print("🔄 Applying intelligent data type conversions...")
        cleaned_df = llm_convert_data_types(cleaned_df, cleaning_strategy, cleaning_report)
    
    # 4. Handle outliers in numeric columns
    if data_profile['numeric_columns'] and "outlier" in cleaning_strategy.lower():
        print("🎯 Handling outliers in numeric columns...")
        cleaned_df = llm_handle_outliers(cleaned_df, data_profile['numeric_columns'], cleaning_strategy, cleaning_report)
    
    # 5. Clean text columns
    if data_profile['text_columns']:
        print("🧹 Cleaning text columns...")
        cleaned_df = llm_clean_text_columns(cleaned_df, data_profile['text_columns'], cleaning_strategy, cleaning_report)
    
    # Generate comprehensive analysis
    analysis = {
        'shape': cleaned_df.shape,
        'columns_cleaned': len(cleaning_report['improvements_made']),
        'data_quality_score': calculate_data_quality_score(cleaned_df),
        'numeric_summary': cleaned_df.select_dtypes(include=[np.number]).describe().to_dict() if data_profile['numeric_columns'] else {},
        'cleaning_success': True
    }
    
    return cleaned_df, analysis

def intelligent_text_clean(text: str, llm: GemmaLLM) -> str:
    """Use LLM to intelligently clean text content."""
    if not isinstance(text, str) or len(text) < 10:
        return str(text)
    
    # For very long texts, use basic cleaning to avoid token limits
    if len(text) > 1000:
        return basic_text_clean(text)
    
    try:
        clean_prompt = f"""Clean and normalize this text content for data analysis:

Text: {text[:500]}

Return the cleaned text with:
- Proper capitalization
- Removed unnecessary characters
- Standardized format
- Preserved meaning

Cleaned text:"""
        
        cleaned = llm(clean_prompt, max_tokens=300)
        return cleaned if cleaned and not cleaned.startswith("❌") else basic_text_clean(text)
    except:
        return basic_text_clean(text)

def basic_text_clean(text: str) -> str:
    """Basic text cleaning fallback."""
    if not isinstance(text, str):
        return str(text)
    
    # Basic cleaning steps
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

def llm_handle_missing_values(df: pd.DataFrame, strategy: str, report: Dict[str, Any]):
    """Handle missing values based on LLM recommendations."""
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['int64', 'float64']:
                if "median" in strategy.lower():
                    df[column] = df[column].fillna(df[column].median())
                    report['improvements_made'].append(f"Filled {column} missing values with median")
                else:
                    df[column] = df[column].fillna(df[column].mean())
                    report['improvements_made'].append(f"Filled {column} missing values with mean")
            elif df[column].dtype == 'object':
                if "mode" in strategy.lower():
                    fill_value = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
                    df[column] = df[column].fillna(fill_value)
                    report['improvements_made'].append(f"Filled {column} missing values with mode")
                else:
                    df[column] = df[column].fillna('Unknown')
                    report['improvements_made'].append(f"Filled {column} missing values with 'Unknown'")
    return df

def llm_convert_data_types(df: pd.DataFrame, strategy: str, report: Dict[str, Any]):
    """Convert data types based on LLM recommendations."""
    # Convert date columns
    for column in df.columns:
        if 'date' in column.lower() or 'time' in column.lower():
            if "date" in strategy.lower() and column in strategy:
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    report['improvements_made'].append(f"Converted {column} to datetime")
                except:
                    pass
    return df

def llm_handle_outliers(df: pd.DataFrame, numeric_columns: List[str], strategy: str, report: Dict[str, Any]):
    """Handle outliers based on LLM recommendations."""
    for column in numeric_columns:
        if "cap" in strategy.lower() or "clip" in strategy.lower():
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            
            outliers_before = len(df[(df[column] < lower) | (df[column] > upper)])
            if outliers_before > 0:
                df[column] = df[column].clip(lower=lower, upper=upper)
                report['improvements_made'].append(f"Capped {outliers_before} outliers in {column}")
    return df

def llm_clean_text_columns(df: pd.DataFrame, text_columns: List[str], strategy: str, report: Dict[str, Any]):
    """Clean text columns based on LLM recommendations."""
    for column in text_columns:
        if "clean" in strategy.lower() and "text" in strategy.lower():
            df[column] = df[column].astype(str).str.strip().str.title()
            report['improvements_made'].append(f"Cleaned and formatted {column}")
    return df

# ========== Utility Functions (same as before) ==========

def handle_missing_values(df, report):
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['int64', 'float64']:
                df[column] = df[column].fillna(df[column].median())
                report['columns_cleaned'][column] = 'Filled missing with median'
            elif df[column].dtype == 'object':
                fill_value = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
                df[column] = df[column].fillna(fill_value)
                report['columns_cleaned'][column] = 'Filled missing with mode/Unknown'
            elif 'date' in column.lower() or 'time' in column.lower():
                df[column] = df[column].fillna(method='ffill')
                report['columns_cleaned'][column] = 'Filled missing dates with forward fill'
    return df


def standardize_data_types(df, report):
    for column in df.columns:
        if any(keyword in column.lower() for keyword in ['date', 'time', 'timestamp']):
            try:
                # Try to infer format from first non-null value to avoid warnings
                sample_value = df[column].dropna().iloc[0] if not df[column].dropna().empty else None
                if sample_value:
                    # Common date formats to try
                    formats_to_try = [
                        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                        '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d', '%d.%m.%Y',
                        '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S'
                    ]
                    
                    # Try each format first
                    format_found = None
                    for fmt in formats_to_try:
                        try:
                            pd.to_datetime([str(sample_value)], format=fmt)
                            format_found = fmt
                            break
                        except:
                            continue
                    
                    if format_found:
                        df[column] = pd.to_datetime(df[column], format=format_found, errors='coerce')
                    else:
                        # Fallback: use infer_datetime_format to reduce warnings
                        df[column] = pd.to_datetime(df[column], errors='coerce', infer_datetime_format=True)
                else:
                    df[column] = pd.to_datetime(df[column], errors='coerce', infer_datetime_format=True)
                    
                report['data_type_changes'][column] = 'Converted to datetime'
            except:
                pass
        elif df[column].dtype == 'object' and df[column].str.isnumeric().all():
            df[column] = pd.to_numeric(df[column], errors='coerce')
            report['data_type_changes'][column] = 'Converted to numeric'
    return df


def clean_string_columns(df, report):
    for column in df.select_dtypes(include=['object']).columns:
        # Skip columns that contain dictionaries (common in PDF/document data)
        if column == 'metadata' or df[column].apply(lambda x: isinstance(x, dict)).any():
            report['columns_cleaned'][column] = 'Skipped - contains dictionary objects'
            continue
            
        # Only process string columns
        try:
            df[column] = df[column].str.strip()
            if any(keyword in column.lower() for keyword in ['name', 'title', 'description']):
                df[column] = df[column].str.title()
            elif any(keyword in column.lower() for keyword in ['code', 'id', 'abbreviation']):
                df[column] = df[column].str.upper()
            df[column] = df[column].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)) if pd.notnull(x) else x)
            report['columns_cleaned'][column] = 'Standardized text formatting'
        except Exception as e:
            report['columns_cleaned'][column] = f'Skipped due to error: {str(e)}'
    return df


def handle_outliers(df, report, method='iqr'):
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower) | (df[column] > upper)]
        if not outliers.empty:
            df[column] = np.where(df[column] < lower, lower, df[column])
            df[column] = np.where(df[column] > upper, upper, df[column])
            report['outliers_handled'][column] = f'Capped {len(outliers)} outliers'
    return df


def validate_data_ranges(df, report):
    rules = {'age': (0, 120), 'salary': (0, 1_000_000), 'rating': (1, 5)}
    for column, (min_val, max_val) in rules.items():
        if column in df.columns:
            invalid_count = len(df[(df[column] < min_val) | (df[column] > max_val)])
            if invalid_count > 0:
                df[column] = df[column].clip(min_val, max_val)
                report['columns_cleaned'][column] = f'Clipped {invalid_count} values to range'
    return df


def create_new_features(df, report):
    for col in [c for c in df.columns if 'date' in c.lower()]:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            new_col = f"{col}_year"
            df[new_col] = df[col].dt.year
            report['columns_cleaned'][new_col] = 'Created year feature from date'
    return df


def generate_cleaning_summary(cleaned_df, original_df):
    return {
        'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
        'columns_remaining': cleaned_df.shape[1],
        'missing_values_removed': original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum(),
        'data_quality_score': calculate_data_quality_score(cleaned_df)
    }


def calculate_data_quality_score(df):
    completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    uniqueness = 1 - (df.duplicated().sum() / df.shape[0])
    return round((completeness * 0.7 + uniqueness * 0.3) * 100, 2)


def perform_comprehensive_analysis(df):
    return {
        'descriptive_stats': df.describe().to_dict(),
        'correlation_matrix': df.select_dtypes(include=[np.number]).corr().to_dict(),
        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'missing_values': df.isnull().sum().to_dict()
    }


def clean_text_document(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)
