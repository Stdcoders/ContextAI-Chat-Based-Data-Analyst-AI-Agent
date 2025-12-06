# data_cleaning_tools.py
import pandas as pd
import numpy as np
import re
import string
import json
from typing import Dict, Any, List, Optional

# LangChain import for creating the tool
from langchain_core.tools import tool

# Your LLM wrapper (must be available in the runtime)
from gemma_llm import GemmaLLM

# Optional NLTK imports for deeper text cleaning
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    _nltk_available = True
except Exception:
    _nltk_available = False
    # Keep function fallbacks below

# -------------------------
# Main exposed LangChain tool
# -------------------------
@tool
def clean_and_analyze_data(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    LLM-driven structured-data cleaning tool.
    - Asks the LLM to produce a strict JSON cleaning plan.
    - Validates and applies the plan safely using pandas.
    - Returns a cleaned DataFrame and a detailed cleaning report.

    Args:
        df (pd.DataFrame): Input DataFrame (must be non-null).
        dataset_name (str): Dataset name for reporting/logging.

    Returns:
        dict: {
            "cleaned_dataframe": pd.DataFrame,
            "cleaning_report": dict
        }
    """
    if df is None:
        raise ValueError("Input DataFrame is None")

    print(f"🧹 Running LLM-driven cleaning on dataset '{dataset_name}'")
    llm = GemmaLLM(temperature=0.2, max_tokens=800)

    cleaning_report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "original_shape": df.shape,
        "cleaning_method": "llm_structured_plan",
        "steps_performed": [],
        "llm_recommendations": {},
        "issues_found": {},
        "improvements_made": [],
        "final_shape": None,
        "final_analysis": {}
    }

    # Build a simple data profile to pass to the LLM
    data_profile = build_data_profile(df)
    cleaning_report['data_profile'] = data_profile

    # Request a structured cleaning plan from LLM
    plan = llm_generate_cleaning_plan(llm, data_profile)
    cleaning_report['llm_recommendations'] = plan

    # Validate plan and apply it
    validated_plan = validate_cleaning_plan(plan, data_profile)
    cleaning_report['validated_plan'] = validated_plan

    cleaned_df = apply_cleaning_plan(df.copy(), validated_plan, cleaning_report)

    # Optional: run type conversions recommended by plan
    if validated_plan.get('convert_types'):
        cleaned_df = llm_convert_data_types(cleaned_df, validated_plan, cleaning_report)

    # Final analysis
        cleaning_report['final_shape'] = cleaned_df.shape
    cleaning_report['cleaning_success'] = True
    cleaning_report['final_analysis'] = {
        'data_quality_score': calculate_data_quality_score(cleaned_df),
        'missing_values_after': cleaned_df.isnull().sum().to_dict(),
        'duplicates_after': int(cleaned_df.duplicated().sum())
    }

    # -----------------------------
    # Generate and store a preview
    # -----------------------------
    try:
        if cleaned_df.shape[0] > 0:
            if "content" in cleaned_df.columns:  # For text/unstructured data
                sample_preview = cleaned_df["content"].head(5).tolist()
                print("\n📄 Sample of cleaned text data:")
                for i, text_line in enumerate(sample_preview, 1):
                    print(f" {i}. {text_line}")
            else:  # For structured/tabular data
                sample_preview = cleaned_df.head(5).to_dict(orient="records")
                print("\n📊 Sample of cleaned tabular data:")
                for i, row in enumerate(sample_preview, 1):
                    print(f" {i}. {row}")
        else:
            sample_preview = []
    except Exception as e:
        sample_preview = []
        print(f"⚠️ Could not generate preview: {e}")

    # Save preview in report for later access by other agents
    cleaning_report["sample_preview"] = sample_preview

    print(f"\n✅ Cleaning completed for '{dataset_name}'. Final shape: {cleaned_df.shape}")
    return {
        "cleaned_dataframe": cleaned_df,
        "cleaning_report": cleaning_report,
        "sample_preview": sample_preview
    }

# -------------------------
# LLM plan generation + validation
# -------------------------
def llm_generate_cleaning_plan(llm: GemmaLLM, data_profile: Dict[str, Any], max_tokens: int = 600) -> dict:
    """
    Ask the LLM to return a strict JSON cleaning plan following a fixed schema.
    If parsing fails, returns a conservative fallback plan.
    """
    prompt = f"""
You are a data-cleaning assistant. Given the dataset profile below, produce a single JSON object (no explanation).
Profile: {json.dumps(data_profile)}

Required schema (return only JSON conforming to this schema):

{{
  "remove_duplicates": true|false,
  "missing_values": {{
      "strategy": "median|mean|mode|constant|drop|leave",
      "constant_values": {{ "<column>": "<value>" }},    # optional
      "numeric_strategy": "median|mean|leave",
      "categorical_strategy": "mode|constant|unknown|leave"
  }},
  "outliers": {{
      "strategy": "clip|drop|leave",
      "method": "iqr|zscore",
      "zscore_threshold": 3.0   # optional
  }},
  "convert_types": {{
      "date_columns": ["colname1", "colname2"],
      "manual": {{ "<column>": "int|float|datetime|category|string" }}
  }},
  "text_cleaning": {{
      "columns": ["col1","col2"],
      "steps": ["lowercase","remove_punct","strip","lemmatize","remove_numbers"]
  }},
  "notes": "optional short string"
}}

If unsure, choose conservative options (do not drop rows). Return only the JSON object.
"""
    try:
        raw = llm(prompt, max_tokens=max_tokens)
    except Exception as e:
        raw = ""
        print(f"⚠️ LLM call failed: {e}")

    try:
        print("🧠 LLM raw output:\n", raw)
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            raise ValueError("LLM response not a JSON object")
    except Exception as e:
        # Conservative fallback plan
        print(f"⚠️ Failed to parse plan from LLM, using fallback. Error: {e}")
        plan = {
            "remove_duplicates": False,
            "missing_values": {
                "strategy": "leave",
                "constant_values": {},
                "numeric_strategy": "median",
                "categorical_strategy": "unknown"
            },
            "outliers": {"strategy": "leave", "method": "iqr"},
            "convert_types": {"date_columns": [], "manual": {}},
            "text_cleaning": {"columns": [], "steps": []},
            "notes": f"fallback_due_to_parse_error: {str(e)}"
        }
    return plan

def validate_cleaning_plan(plan: dict, profile: Dict[str, Any]) -> dict:
    """
    Ensure required keys exist, sanitize allowed values, and coerce to safe defaults.
    This prevents unexpected or malicious values from the LLM.
    """
    # Minimal schema with allowed values
    validated = {
        "remove_duplicates": bool(plan.get("remove_duplicates", False)),
        "missing_values": {
            "strategy": "leave",
            "constant_values": {},
            "numeric_strategy": "median",
            "categorical_strategy": "unknown"
        },
        "outliers": {"strategy": "leave", "method": "iqr", "zscore_threshold": 3.0},
        "convert_types": {"date_columns": [], "manual": {}},
        "text_cleaning": {"columns": [], "steps": []},
        "notes": plan.get("notes", "")
    }

    # missing_values
    mv = plan.get("missing_values", {})
    if isinstance(mv, dict):
        strat = mv.get("strategy", "").lower()
        if strat in {"median", "mean", "mode", "constant", "drop", "leave"}:
            validated["missing_values"]["strategy"] = strat
        # numeric / categorical specifics
        nstr = mv.get("numeric_strategy", "").lower()
        if nstr in {"median", "mean", "leave"}:
            validated["missing_values"]["numeric_strategy"] = nstr
        cstr = mv.get("categorical_strategy", "").lower()
        if cstr in {"mode", "constant", "unknown", "leave"}:
            validated["missing_values"]["categorical_strategy"] = cstr
        consts = mv.get("constant_values", {})
        if isinstance(consts, dict):
            validated["missing_values"]["constant_values"] = consts

    # outliers
    out = plan.get("outliers", {})
    if isinstance(out, dict):
        strat = out.get("strategy", "").lower()
        if strat in {"clip", "drop", "leave"}:
            validated["outliers"]["strategy"] = strat
        method = out.get("method", "").lower()
        if method in {"iqr", "zscore"}:
            validated["outliers"]["method"] = method
        if isinstance(out.get("zscore_threshold"), (int, float)):
            validated["outliers"]["zscore_threshold"] = float(out.get("zscore_threshold"))

    # convert_types
    conv = plan.get("convert_types", {})
    if isinstance(conv, dict):
        date_cols = conv.get("date_columns", [])
        if isinstance(date_cols, list):
            # keep only columns that actually exist in profile
            validated["convert_types"]["date_columns"] = [c for c in date_cols if c in profile.get("columns", [])]
        manual = conv.get("manual", {})
        if isinstance(manual, dict):
            # sanitize manual types
            allowed = {"int", "float", "datetime", "category", "string"}
            safe_manual = {c: t for c, t in manual.items() if c in profile.get("columns", []) and t in allowed}
            validated["convert_types"]["manual"] = safe_manual

    # text_cleaning
    txt = plan.get("text_cleaning", {})
    if isinstance(txt, dict):
        cols = txt.get("columns", [])
        steps = txt.get("steps", [])
        if isinstance(cols, list):
            validated["text_cleaning"]["columns"] = [c for c in cols if c in profile.get("columns", [])]
        if isinstance(steps, list):
            # Allow only known steps
            allowed_steps = {"lowercase", "remove_punct", "strip", "lemmatize", "remove_numbers"}
            validated["text_cleaning"]["steps"] = [s for s in steps if s in allowed_steps]

    # remove_duplicates
    validated["remove_duplicates"] = bool(plan.get("remove_duplicates", False))

    return validated

# -------------------------
# Apply the plan safely
# -------------------------
def apply_cleaning_plan(df: pd.DataFrame, plan: dict, report: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply validated cleaning plan using safe pandas operations.
    Each action is logged into the report['improvements_made'] and report['steps_performed'].
    """
    report.setdefault('improvements_made', [])
    report.setdefault('steps_performed', [])

    # 1) Remove duplicates
    if plan.get("remove_duplicates", False):
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            df = df.drop_duplicates()
            report['improvements_made'].append(f"Removed {dup_count} duplicate rows.")
        report['steps_performed'].append("removed_duplicates" if dup_count > 0 else "no_duplicates_found")

    # 2) Missing values
    mv = plan.get("missing_values", {})
    if mv.get("strategy", "leave") != "leave":
        # Numeric columns handling
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if mv.get("numeric_strategy") == "median" or mv.get("strategy") == "median":
                    fill = df[col].median()
                    df[col] = df[col].fillna(fill)
                    report['improvements_made'].append(f"Filled missing numeric '{col}' with median ({fill}).")
                elif mv.get("numeric_strategy") == "mean" or mv.get("strategy") == "mean":
                    fill = df[col].mean()
                    df[col] = df[col].fillna(fill)
                    report['improvements_made'].append(f"Filled missing numeric '{col}' with mean ({fill}).")
                elif mv.get("numeric_strategy") == "leave":
                    report['steps_performed'].append(f"left_missing_numeric_{col}")
            else:
                # categorical/text
                if mv.get("categorical_strategy") == "mode" or mv.get("strategy") == "mode":
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    report['improvements_made'].append(f"Filled missing categorical '{col}' with mode ('{mode_val}').")
                elif mv.get("categorical_strategy") == "constant" or mv.get("strategy") == "constant":
                    consts = mv.get("constant_values", {})
                    fill = consts.get(col, "Unknown")
                    df[col] = df[col].fillna(fill)
                    report['improvements_made'].append(f"Filled missing categorical '{col}' with constant ('{fill}').")
                elif mv.get("categorical_strategy") == "unknown":
                    df[col] = df[col].fillna("Unknown")
                    report['improvements_made'].append(f"Filled missing categorical '{col}' with 'Unknown'.")
                elif mv.get("categorical_strategy") == "leave":
                    report['steps_performed'].append(f"left_missing_categorical_{col}")
    else:
        report['steps_performed'].append("missing_values_left")

    # 3) Outliers
    out = plan.get("outliers", {})
    if out.get("strategy", "leave") != "leave":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if out.get("method") == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                mask = (df[col] < lower) | (df[col] > upper)
                outliers_before = int(mask.sum())
                if outliers_before > 0:
                    if out.get("strategy") == "clip":
                        df[col] = df[col].clip(lower=lower, upper=upper)
                        report['improvements_made'].append(f"Capped {outliers_before} outliers in '{col}' using IQR clipping.")
                    elif out.get("strategy") == "drop":
                        df = df[~mask]
                        report['improvements_made'].append(f"Dropped {outliers_before} rows with outliers in '{col}'.")
            elif out.get("method") == "zscore":
                threshold = float(out.get("zscore_threshold", 3.0))
                col_std = df[col].std()
                if col_std == 0 or np.isnan(col_std):
                    continue
                zscores = (df[col] - df[col].mean()) / col_std
                mask = zscores.abs() > threshold
                outliers_before = int(mask.sum())
                if outliers_before > 0:
                    if out.get("strategy") == "clip":
                        # clip to mean +/- threshold*std
                        mean = df[col].mean()
                        lower, upper = mean - threshold * col_std, mean + threshold * col_std
                        df[col] = df[col].clip(lower=lower, upper=upper)
                        report['improvements_made'].append(f"Capped {outliers_before} outliers in '{col}' using zscore clipping.")
                    elif out.get("strategy") == "drop":
                        df = df[~mask]
                        report['improvements_made'].append(f"Dropped {outliers_before} rows with outliers in '{col}' using zscore.")
        report['steps_performed'].append(f"handled_outliers_{out.get('method')}_{out.get('strategy')}")
    else:
        report['steps_performed'].append("outliers_left")

    # 4) Text cleaning
    txt = plan.get("text_cleaning", {})
    tcols = txt.get("columns", [])
    steps = txt.get("steps", [])
    if tcols and steps:
        for col in tcols:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str)
            # Apply in safe, deterministic order
            if "strip" in steps:
                df[col] = df[col].str.strip()
            if "lowercase" in steps:
                df[col] = df[col].str.lower()
            if "remove_punct" in steps:
                df[col] = df[col].apply(lambda s: re.sub(r'[^\w\s]', '', s))
            if "remove_numbers" in steps:
                df[col] = df[col].apply(lambda s: re.sub(r'\d+', '', s))
            if "lemmatize" in steps and _nltk_available:
                lemmatizer = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))
                def _lemmatize_text(t):
                    tokens = re.findall(r'\w+', t)
                    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
                    return " ".join(lemmatizer.lemmatize(w) for w in tokens)
                df[col] = df[col].apply(_lemmatize_text)
            report['improvements_made'].append(f"Applied text cleaning on '{col}': {steps}")
        report['steps_performed'].append("text_cleaning_done")
    else:
        report['steps_performed'].append("text_cleaning_skipped")

    # 5) Convert types as per validated plan (manual conversions handled later by llm_convert_data_types)
    # We only validate that conversions are recorded; actual conversions occur in llm_convert_data_types
    report['steps_performed'].append("apply_plan_complete")
    return df

# -------------------------
# Type conversion & helpers
# -------------------------
def llm_convert_data_types(df: pd.DataFrame, plan: dict, report: Dict[str, Any]) -> pd.DataFrame:
    """
    Perform safe type conversions recorded in plan['convert_types'].
    """
    conv = plan.get("convert_types", {})
    # Date columns
    date_cols = conv.get("date_columns", [])
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                report['improvements_made'].append(f"Converted '{col}' to datetime.")
            except Exception:
                report['improvements_made'].append(f"Failed to convert '{col}' to datetime (kept original).")

    # Manual conversions
    manual = conv.get("manual", {})
    for col, dtype in manual.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                report['improvements_made'].append(f"Converted '{col}' to int (nullable Int64).")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                report['improvements_made'].append(f"Converted '{col}' to float.")
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
                report['improvements_made'].append(f"Converted '{col}' to datetime.")
            elif dtype == "category":
                df[col] = df[col].astype('category')
                report['improvements_made'].append(f"Converted '{col}' to category.")
            elif dtype == "string":
                df[col] = df[col].astype(str)
                report['improvements_made'].append(f"Converted '{col}' to string.")
        except Exception:
            report['improvements_made'].append(f"Failed to convert '{col}' to {dtype} (kept original).")
    report['steps_performed'].append("type_conversions_done")
    return df

# -------------------------
# Utility helpers from original file (kept / improved)
# -------------------------
def basic_text_clean(text: str) -> str:
    """Basic text cleaning fallback."""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

def clean_text_document(text: str) -> str:
    """Performs deep cleaning on a string of text using NLTK if available."""
    if not isinstance(text, str):
        return str(text)
    if not _nltk_available:
        return basic_text_clean(text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = re.findall(r'\w+', text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def llm_handle_missing_values(df: pd.DataFrame, strategy: str, report: Dict[str, Any]):
    """Fallback legacy helper (kept for compatibility)."""
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[column]):
                fill_val = df[column].median()
                df[column] = df[column].fillna(fill_val)
                report['improvements_made'].append(f"Filled missing '{column}' with median ({fill_val}).")
            else:
                fill_val = 'Unknown'
                df[column] = df[column].fillna(fill_val)
                report['improvements_made'].append(f"Filled missing '{column}' with '{fill_val}'")
    return df

def llm_convert_data_types_legacy(df: pd.DataFrame, strategy: str, report: Dict[str, Any]):
    """Legacy typed conversion (kept for reference)."""
    return df

def llm_handle_outliers(df: pd.DataFrame, numeric_columns: List[str], strategy: str, report: Dict[str, Any]):
    """Legacy outlier handler (kept for reference)."""
    return df

def llm_clean_text_columns(df: pd.DataFrame, text_columns: List[str], strategy: str, report: Dict[str, Any]):
    """Legacy text cleaning wrapper (kept for reference)."""
    return df

def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """Calculates a simple data quality score (0-100)."""
    if df is None or df.empty:
        return 0.0
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    completeness = 1 - (missing / total_cells) if total_cells > 0 else 1.0
    uniqueness = 1 - (df.duplicated().sum() / df.shape[0]) if df.shape[0] > 0 else 1.0
    return round((completeness * 0.7 + uniqueness * 0.3) * 100, 2)

def build_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a lightweight profile for the LLM prompt (kept concise)."""
    profile = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "sample_values": {c: _sample_values(df[c]) for c in df.columns[:5]}  # limited sample to keep prompt small
    }
    return profile

def _sample_values(series: pd.Series, n: int = 3):
    try:
        return list(series.dropna().astype(str).unique()[:n])
    except Exception:
        return []

# End of file
