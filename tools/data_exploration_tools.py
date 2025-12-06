import json
import re
import numpy as np
from typing import Dict, List

# LangChain import for creating the tool
from langchain_core.tools import tool

# Your existing imports (make sure groq_llm.py is accessible)
from groq_llm import GroqLLM

# ================== TOOL 1: DATA UNDERSTANDING ==================

@tool
def understand_data(df_profile: dict) -> dict:
    """
    Uses an LLM to semantically understand a dataset's domain, purpose, and context
    based on its profile (column names, data types, sample content, etc.).
    It intelligently routes between analyzing structured data and unstructured text.
    
    Args:
        df_profile (dict): A dictionary containing the profile of the dataset,
                           including keys like 'data_type', 'columns', and 'sample'.
                           
    Returns:
        dict: A dictionary containing the analysis, including the detected 'domain'.
    """
    print("🧠 Tool 'understand_data' starting...")
    data_type = df_profile.get('data_type', 'unknown')
    
    # Route to the appropriate helper based on data type
    if data_type in ['pdf', 'text']:
        understanding = _text_document_understanding(df_profile)
    else:
        understanding = _structured_data_understanding(df_profile)

    print(f"✅ Tool 'understand_data' finished. Detected Domain: {understanding.get('domain', 'General')}")
    return understanding


# ================== TOOL 2: QUESTION GENERATION ==================

@tool
def generate_questions(df_profile: dict, understanding: dict, num_questions: int = 8) -> list:
    """
    Generates a set of insightful analytical questions based on the dataset profile and
    the domain understanding provided by the 'understand_data' tool.
    
    Args:
        df_profile (dict): The profile of the dataset.
        understanding (dict): The output from the 'understand_data' tool, including the 'domain'.
        num_questions (int): The number of questions to generate.
        
    Returns:
        list: A list of generated question strings.
    """
    print(f"❓ Tool 'generate_questions' starting for domain: {understanding.get('domain', 'Unknown')}")
    agent = QuestionGenerationAgent()
    questions = agent.generate(df_profile, understanding, num_questions)
    
    print(f"✅ Tool 'generate_questions' finished. Generated {len(questions)} questions.")
    return questions


# ========== INTERNAL LOGIC AND HELPER CLASSES/FUNCTIONS ==========
# This section contains all the original helper functions and classes.
# They are not tools themselves but are used internally by the tools above.

def _text_document_understanding(df_profile: dict) -> dict:
    """Helper to analyze text documents based on content."""
    try:
        llm = GroqLLM(temperature=0.3, max_tokens=300)
        sample_content = ""
        if 'sample' in df_profile and df_profile['sample']:
            for sample in df_profile['sample'][:3]:
                if isinstance(sample, dict) and 'content' in sample:
                    sample_content += sample['content'][:500] + " "
        sample_content = sample_content[:1500]

        prompt = f"Analyze the following document content to determine its domain, type, and main topics.\n\nContent:\n{sample_content}"
        if llm.is_available():
            response_text = llm(prompt)
            domain = _extract_domain_from_content(sample_content, response_text)
            return {"domain": domain, "analysis": response_text}
        return _fallback_text_understanding(df_profile, sample_content)
    except Exception as e:
        print(f"Warning: LLM analysis for text understanding failed: {e}")
        return _fallback_text_understanding(df_profile, "")


def _structured_data_understanding(df_profile: dict) -> dict:
    """Helper to analyze structured data based on column names."""
    try:
        llm = GroqLLM(temperature=0.3, max_tokens=300)
        column_names = [col.get('name', 'Unknown') for col in df_profile.get('columns', [])[:10]]
        prompt = f"Based ONLY on these column names, what is the likely domain and purpose of this dataset? Be concise. Columns: {column_names}"

        if llm.is_available():
            response_text = llm(prompt)
            domain_match = re.search(r'Domain[:\s]*([^\n,]+)', response_text, re.IGNORECASE)
            domain = domain_match.group(1).strip() if domain_match else "General"
            return {"domain": domain, "analysis": response_text}
        return _fallback_understanding(df_profile)
    except Exception as e:
        print(f"Warning: LLM analysis for structured understanding failed: {e}")
        return _fallback_understanding(df_profile)


def _extract_domain_from_content(content: str, llm_response: str) -> str:
    """Extracts domain from document content and LLM response using keywords."""
    # First try to extract from LLM response
    domain_match = re.search(r'Domain[:\s]*([^\n,]+)', llm_response, re.IGNORECASE)
    if domain_match:
        return domain_match.group(1).strip()
        
    # Your comprehensive keyword-based domain extraction logic
    content_lower = content.lower()
    domain_keywords = {
        "Technology/AI/ML": ['machine learning', 'algorithm', 'software', 'data science'],
        "Academic/Research": ['research', 'study', 'methodology', 'findings'],
        "Business/Finance": ['business', 'market', 'revenue', 'sales', 'finance'],
        "Healthcare/Medical": ['health', 'medical', 'patient', 'treatment', 'clinical'],
    }
    for domain, keywords in domain_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            return domain
    return "General"


def _fallback_text_understanding(df_profile: dict, content: str) -> dict:
    """Fallback text analysis when LLM is not available."""
    domain = _extract_domain_from_content(content, "")
    return {
        "domain": domain,
        "analysis": f"Fallback analysis suggests a {domain} domain based on keywords.",
        "fallback": True
    }


def _fallback_understanding(df_profile: dict) -> dict:
    """Fallback analysis for structured data when LLM is not available."""
    columns = df_profile.get('columns', [])
    column_names = [str(col.get('name', '')).lower() for col in columns]
    all_columns_str = ' '.join(column_names)
    
    if any(word in all_columns_str for word in ['sleep', 'duration', 'quality', 'bedtime']):
        domain = "Sleep/Health"
    elif any(word in all_columns_str for word in ['patient', 'diagnosis', 'treatment', 'medical']):
        domain = "Healthcare/Medical"
    elif any(word in all_columns_str for word in ['price', 'cost', 'revenue', 'sales']):
        domain = "Business/Finance"
    else:
        domain = "General"
        
    return {
        "domain": domain,
        "analysis": f"Fallback analysis suggests {domain} domain based on column names.",
        "fallback": True
    }


class QuestionGenerationAgent:
    """Internal agent class to handle the logic of generating questions."""
    def __init__(self, model: str = "gemini-1.5-flash-latest"):
        self.model_name = model
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        try:
            self.llm = GroqLLM(temperature=0.8, max_tokens=1500)
            if not self.llm.is_available():
                print("Warning: GroqLLM is not available for QuestionGenerationAgent.")
                self.llm = None
        except Exception as e:
            print(f"Warning: Failed to initialize GroqLLM for QuestionGenerationAgent: {e}")
            self.llm = None

    def generate(self, profile: dict, understanding: dict, num_questions: int = 10):
        data_type = profile.get('data_type', 'unknown')
        if data_type in ['pdf', 'text']:
            return self._generate_text_questions(profile, understanding, num_questions)
        
        if self.llm is None:
            return self._fallback_questions(profile, num_questions)

        columns = profile.get('columns', [])
        col_entries = [f"{col.get('name', '')} ({col.get('dtype', '')})" for col in columns[:20]]
        domain = understanding.get('domain', 'General')
        
        prompt = (
            f"You are a data analyst. For a '{domain}' dataset with columns: {', '.join(col_entries)}, "
            f"generate {num_questions} specific, data-driven analytical questions. "
            "Output ONLY a JSON array of strings (the questions), with no other text or explanation."
        )
        try:
            response_text = self.llm(prompt)
            # Clean the response to find the JSON array
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group(0))
                return questions[:num_questions]
            else:
                 raise ValueError("No JSON array found in response")
        except (Exception, json.JSONDecodeError):
            raw_questions = re.findall(r'\d+\.\s*(.+)', response_text)
            return raw_questions[:num_questions] if raw_questions else self._fallback_questions(profile, num_questions)

    def _generate_text_questions(self, profile: dict, understanding: dict, num_questions: int) -> list:
        if self.llm:
            sample_content = self._extract_text_content(profile)
            prompt = f"Generate {num_questions} insightful questions about the following document. Output only a numbered list of questions.\n\n{sample_content[:1500]}"
            response = self.llm(prompt)
            return re.findall(r'\d+\.\s*(.+)', response)
        return self._fallback_text_questions(profile, "", num_questions)

    def _extract_text_content(self, profile: dict) -> str:
        sample_content = ""
        if 'sample' in profile and profile['sample']:
            for sample in profile['sample'][:3]:
                if isinstance(sample, dict) and 'content' in sample and isinstance(sample['content'], str):
                    sample_content += sample['content'][:600].strip() + "\n\n"
        return sample_content.strip() if sample_content else "No content available."

    def _fallback_text_questions(self, profile: dict, content: str, num_questions: int) -> list:
        return [
            "What are the main topics and themes discussed in this document?",
            "What are the key conclusions or findings presented in the text?",
            "What is the primary purpose or objective of this document?",
            "Who is the intended audience for this text?",
            "What are the most significant insights that can be drawn from the content?",
        ][:num_questions]

    def _fallback_questions(self, profile: dict, num_questions: int) -> list:
        questions = ["What are the key descriptive statistics of the dataset?"]
        columns = profile.get('columns', [])
        if columns:
            numeric_cols = [c.get('name') for c in columns if 'int' in c.get('dtype', '') or 'float' in c.get('dtype', '')]
            cat_cols = [c.get('name') for c in columns if 'object' in c.get('dtype', '')]

            if cat_cols:
                questions.append(f"What is the distribution of categories in '{cat_cols[0]}'?")
            if numeric_cols:
                questions.append(f"What is the distribution of '{numeric_cols[0]}'?")
            if len(numeric_cols) > 1:
                 questions.append(f"What is the relationship between '{numeric_cols[0]}' and '{numeric_cols[1]}'?")
            if numeric_cols and cat_cols:
                questions.append(f"How does the average of '{numeric_cols[0]}' vary by '{cat_cols[0]}'?")

        return questions[:num_questions]