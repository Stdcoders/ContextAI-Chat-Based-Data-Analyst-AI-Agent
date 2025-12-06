import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json

# LangChain import for creating the tool
from langchain_core.tools import tool

# --- Import all required specialized modules ---
try:
    from nodes.llm_question_classifier import LLMQuestionClassifier, QuestionIntent
    from nodes.hybrid_calculation_engine import HybridCalculationEngine
    from gemma_llm import GemmaLLM
    from groq_llm import GroqLLM
except ImportError:
    print("FATAL ERROR: Could not import required modules. Please ensure all tool files are in the correct path.")
    # Provide minimal fallbacks for basic script loading
    class LLMQuestionClassifier:
        def analyze_question_intent(self, q, df_context): return type('obj', (object,), {'intent': 'descriptive', 'confidence': 0, 'reasoning': 'Fallback'})()
    class HybridCalculationEngine:
        def analyze_with_calculations(self, q, df): return "Hybrid Engine not available."
    class GemmaLLM:
        def __init__(self, **kwargs): pass
        def is_available(self): return False
        def __call__(self, prompt): return "Gemma LLM not available."
    # --- THIS IS THE FIX ---
    # The 'QuestionIntent' class was previously nested inside itself, causing the error.
    # It has been corrected here by removing the extra nested class definition.
    class QuestionIntent:
        STATISTICAL, COMPARATIVE, ANALYTICAL, DESCRIPTIVE, FORECASTING = "statistical", "comparative", "analytical", "descriptive", "forecasting"

# Optional Plotly imports for visualization
try:
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objs as go
except ImportError:
    px, pio, go = None, None, None
    print("Warning: Plotly not found. Visualization will be disabled.")

# Optional Prophet import for forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None # Define Prophet as None if not available

# ================== THE LANGCHAIN TOOL ==================
@tool
def analyze_user_question(question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Answers a user's question about the dataset by using an LLM to classify the question's
    intent and then routing it to the appropriate specialized analysis engine for calculation,
    visualization, or descriptive answers. This is the primary tool for all data analysis queries.
    """
    print(f"🔬 Tool 'analyze_user_question' starting for question: '{question}'")
    try:
        engine = UnifiedAnalysisEngine()
        result = engine.analyze_question(question, df, dataset_name)
        print("✅ Tool 'analyze_user_question' finished successfully.")
        return result
    except Exception as e:
        print(f"❌ Error in 'analyze_user_question' tool: {e}")
        return {"error": str(e), "answer": f"An error occurred during analysis: {e}"}

# ========== INTERNAL CLASSES AND LOGIC ==========

class InsightAgent:
    """
    Specialized agent for generating descriptive text answers and LLM-powered visualizations.
    """
    def __init__(self):
        self.desc_llm = GemmaLLM(model_name="google/gemma-2-9b-it", temperature=0.2, max_tokens=1500)
        self.viz_llm = GroqLLM(model_name="openai/gpt-oss-120b", temperature=0.0, max_tokens=1000)
        # Unified LLM provider for internal tasks
        self.llm_provider = GroqLLM(model_name="openai/gpt-oss-120b", temperature=0.0)


    def generate_descriptive_answer(self, question: str, df: pd.DataFrame, dataset_name: str) -> str:
        if not self.desc_llm.is_available(): return "Insight LLM is not available."
        context = (f"Dataset: '{dataset_name}'\nShape: {df.shape}\nColumns: {df.columns.tolist()}\nHead:\n{df.head(3).to_markdown()}")
        prompt = (f"Context:\n{context}\n\nUser Question: \"{question}\"\n\nAnswer the question based only on the context provided.")
        return self.desc_llm(prompt)
    
    def intelligent_data_cleaning(self, df: pd.DataFrame, question: str) -> pd.DataFrame:
        """
    Uses the LLM to automatically detect and clean numeric or GDP-like columns,
    fixing formatting issues such as commas, currency symbols, or units.
    """
        if not self.llm_provider.is_available():
            print("❌ LLM not available for intelligent data cleaning.")
            return df  # fallback

        print("🧠 Using LLM for intelligent numeric cleaning...")

        df_preview = df.head(3).to_string()
        df_dtypes = df.dtypes.to_string()

        prompt = f"""
    You are a data preprocessing expert. The following pandas DataFrame represents an economic dataset.

    QUESTION: "{question}"

    DATA PREVIEW:
    {df_preview}

    DATA TYPES:
    {df_dtypes}

    TASK:
    1. Identify columns that represent numerical quantities but are stored as text (e.g., GDP values like "12,345 USD" or "1.2 million").
    2. Describe how to clean them (e.g., remove commas, symbols, words, convert to float).
    3. Generate executable Python code that converts these columns to numeric using pandas.

    Your response must ONLY contain valid Python code that modifies the DataFrame `df` in place.
    Example:
    df['GDP'] = df['GDP'].replace(',', '', regex=True).replace(r'[^0-9.\-]', '', regex=True).astype(float)
    """

        try:
            llm_response = self.llm_provider(prompt)
            code = llm_response.strip().replace("```python", "").replace("```", "")
            print("✅ LLM Cleaning Code Generated:\n", code)
            exec(code, {"df": df, "pd": pd, "np": np})
            return df
        except Exception as e:
            print(f"❌ Intelligent cleaning failed: {e}")
            return df


    def _get_forecasting_parameters(self, question: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Uses an LLM to identify the date column, value column, and forecast period
        from a user's question and the dataframe's context.
        """
        print("🧠 Using LLM to determine forecasting parameters...")
        if not self.llm_provider.is_available():
            print("❌ LLM provider not available for parameter extraction.")
            return None

        df_head_str = df.head().to_string()
        df_dtypes_str = df.dtypes.to_string()

        prompt = f"""
        Analyze the user's question and the provided DataFrame context to identify the parameters for a time-series forecast.

        **User Question:** "{question}"

        **DataFrame Head:**
        {df_head_str}

        **DataFrame Columns and Data Types:**
        {df_dtypes_str}

        **Your Task:**
        1.  **Identify the Date Column:** Find the column that most likely represents the date or time. Look for columns with names like 'date', 'timestamp', 'time', or of a datetime data type.
        2.  **Identify the Value Column:** Find the column that contains the numerical data to be forecasted, based on the user's question (e.g., 'sales', 'revenue', 'users').
        3.  **Identify the Forecast Period:** Determine the number of days to forecast into the future. Infer from terms like "next month" (30 days), "next year" (365 days), "next quarter" (90 days). If not specified, default to 90 days.

        **Output Format:**
        Provide your answer ONLY as a valid JSON object with the following keys:
        - "date_column": string (the name of the date column)
        - "value_column": string (the name of the value column)
        - "periods": integer (the number of days to forecast)

        Example Output:
        {{
            "date_column": "OrderDate",
            "value_column": "Sales",
            "periods": 365
        }}
        """

        try:
            response = self.llm_provider(prompt)
            cleaned_response = response.strip().replace('`', '').replace('json', '')
            params = json.loads(cleaned_response)

            if params['date_column'] not in df.columns or params['value_column'] not in df.columns:
                print(f"❌ LLM identified invalid columns: {params}. Aborting.")
                return None

            print(f"✅ LLM identified parameters: {params}")
            return params
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"❌ Error parsing LLM response for forecast parameters: {e}")
            return None

    def forecast_time_series(self, df: pd.DataFrame, date_column: str, value_column: str, periods: int) -> Dict[str, Any]:
        """
        Performs time-series forecasting using Prophet and returns the result.
        """
        if not PROPHET_AVAILABLE:
            return {"answer": "Prophet library not installed. Please run 'pip install prophet' to enable forecasting.", "visualization": None}

        print(f"📊 Forecasting {value_column} for the next {periods} days...")
        prophet_df = df[[date_column, value_column]].rename(columns={
            date_column: 'ds',
            value_column: 'y'
        })
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty',
            mode='lines', line_color='rgba(255,255,255,0)', name='Confidence Upper'
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty',
            mode='lines', line_color='rgba(255,255,255,0)', name='Confidence Lower'
        ))
        fig.update_layout(title=f'Forecast of {value_column}', xaxis_title='Date', yaxis_title=value_column)
        viz = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

        last_forecast_value = forecast.iloc[-1]['yhat']
        answer = (
            f"Forecasting complete. The forecast for '{value_column}' extends to {forecast['ds'].max().strftime('%Y-%m-%d')}. "
            f"The predicted value on that date is approximately {last_forecast_value:,.2f}."
        )

        return {"answer": answer, "visualization": viz}

    def create_visualization(self, df: pd.DataFrame, question: str):
        if not px or not self.viz_llm.is_available(): return None, None

        prompt = self._create_viz_prompt(df, question)
        viz_code = self.viz_llm(prompt)
        cleaned_code = self._clean_generated_code(viz_code)

        if not cleaned_code or '(' not in cleaned_code or ')' not in cleaned_code:
            print(f"❌ LLM returned incomplete code: '{cleaned_code}'.")
            return None, None
            
        print(f"✅ Generated Plotly Code for '{question}': `{cleaned_code}`")

        try:
            scope = {"df": df, "px": px, "pd": pd, "np": np}
            fig = eval(cleaned_code, scope)
            
            if fig and hasattr(fig, 'to_html'):
                return fig.to_html(full_html=False, include_plotlyjs='cdn'), pio.to_json(fig)
            else:
                print("❌ Executed code did not produce a valid Plotly figure.")
        except Exception as e:
            print(f"❌ Visualization execution failed for '{question}': {e}\n  Code: {cleaned_code}")
        return None, None

    def _create_viz_prompt(self, df: pd.DataFrame, question: str) -> str:
        col_details = "\n".join([f"- '{col}' (type: {dtype})" for col, dtype in df.dtypes.items()])
        return f"""
        You are an expert data visualization specialist. Given a pandas DataFrame named 'df', write a single, executable line of Python code using Plotly Express (aliased as `px`) to create an attractive and informative plot that best answers the user's question.

        USER QUESTION: "{question}"

        DATAFRAME COLUMNS:
        {col_details}

        VISUALIZATION RULES:
        1.  **CHOOSE THE BEST PLOT:** Select the most appropriate plot type (e.g., bar, scatter, box, histogram, line).
        2.  **ADD A CLEAR TITLE:** Always include a descriptive `title`.
        3.  **IMPROVE AXIS LABELS:** Use the `labels` parameter to make axis titles more readable.
        4.  **USE A PROFESSIONAL TEMPLATE:** Set `template='plotly_dark'`.
        5.  **USE COLOR INTELLIGENTLY:** If a third variable is relevant, use the `color` parameter.
        6.  **FINAL OUTPUT:** The output must be a single line of Python code. Do not include `fig =`, explanations, or markdown.

        EXAMPLE:
        User Question: "Show the relationship between charges and bmi for smokers vs non-smokers"
        Code: px.scatter(df, x='bmi', y='charges', color='smoker', title='Charges vs. BMI by Smoking Status', labels={{'bmi': 'Body Mass Index', 'charges': 'Insurance Charges ($)'}}, template='plotly_dark')

        CODE:
        """.strip()

    def _clean_generated_code(self, code: str) -> str:
        code = str(code).strip().replace("`python", "").replace("`", "")
        return code[6:] if code.startswith("fig = ") else code

class UnifiedAnalysisEngine:
    def __init__(self):
        self.classifier = LLMQuestionClassifier()
        self.hybrid_engine = HybridCalculationEngine()
        self.insight_agent = InsightAgent()

    def analyze_question(self, question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
    Routes user questions to the appropriate analysis module (EDA, statistical, descriptive, forecasting),
    while applying LLM-driven intelligent data cleaning when necessary.
    """
    # --- Identify visualization-related requests ---
        visualization_keywords = ['plot', 'visualize', 'chart', 'graph', 'histogram', 'scatter', 'bar', 'line']
        is_visualization_request = any(keyword in question.lower() for keyword in visualization_keywords)

    # --- Classify the intent using the LLM ---
        df_context = f"Dataset has columns: {df.columns.tolist()}"
        classification = self.classifier.analyze_question_intent(question, df_context)
        intent_value = getattr(classification.intent, 'value', 'descriptive')

        print(f"🎯 LLM Classification: {intent_value.upper()} (Confidence: {classification.confidence:.1%})")
        print(f"   Reasoning: {classification.reasoning}")

    # --- Intelligent Data Cleaning ---
    # Run LLM-powered cleaning before any analysis requiring numerical computation or visualization
        if is_visualization_request or intent_value in ["statistical", "comparative", "analytical"]:
            print("🧹 Initiating intelligent data cleaning before analysis...")
            df = self.insight_agent.intelligent_data_cleaning(df, question)

    # --- ROUTING LOGIC ---
        if is_visualization_request or intent_value in ["statistical", "comparative", "analytical"]:
            return self._handle_statistical_question(question, df, classification)

        if "eda" in question.lower():
        # Clean before EDA too (optional)
            print("🧹 Running LLM-based cleaning for EDA...")
            df = self.insight_agent.intelligent_data_cleaning(df, question)
            return self._handle_eda_question(question, df, classification)

        elif intent_value == "forecasting":
            params = self.insight_agent._get_forecasting_parameters(question, df)
            if params:
                result = self.insight_agent.forecast_time_series(
                df=df,
                date_column=params['date_column'],
                value_column=params['value_column'],
                periods=params['periods']
            )
                final_result = result.copy()
                final_result['method'] = 'LLM-Classified (forecasting) -> Forecasting Tool'
                return final_result
            else:
                return {
                'answer': "I couldn't determine the correct parameters to run a forecast. Please be more specific.",
                'visualization': None,
                'method': 'forecasting_parameter_failure'
            }

        else:  # Default to descriptive route
            return self._handle_descriptive_question(question, df, dataset_name, classification)


    def _handle_statistical_question(self, question: str, df: pd.DataFrame, classification) -> Dict[str, Any]:
        print(f"⚡ Using Hybrid Calculation Engine for: {classification.intent}")
        calculation_result = self.hybrid_engine.analyze_with_calculations(question, df)
        viz_html, viz_json = self.insight_agent.create_visualization(df, question)
        return {
            'question': question, 'answer': calculation_result, 'visualization_html': viz_html,
            'visualization_json': viz_json, 'method': f'LLM-Classified ({classification.intent}) -> Hybrid Engine'
        }

    def _handle_eda_question(self, question: str, df: pd.DataFrame, classification) -> Dict[str, Any]:
        print("🧭 Performing comprehensive Exploratory Data Analysis (EDA)...")
        eda_script_question = "Perform a comprehensive EDA. Provide dataset shape, missing values, descriptive statistics for numeric columns, and the top 5 absolute correlations between numeric variables."
        summary_result = self.hybrid_engine.analyze_with_calculations(eda_script_question, df)
        viz_questions = self._generate_eda_viz_questions(df)
        print(f"✅ EDA summary complete. Generating {len(viz_questions)} targeted visualizations...")
        all_viz_html = ""
        first_viz_json = None
        for i, viz_q in enumerate(viz_questions):
            html, json_data = self.insight_agent.create_visualization(df, viz_q)
            if html:
                all_viz_html += f"<div><h2>{viz_q}</h2>{html}</div><hr>"
                if i == 0: first_viz_json = json_data
        return {
            'question': question, 'answer': summary_result, 'visualization_html': all_viz_html,
            'visualization_json': first_viz_json, 'method': f'LLM-Classified (EDA) -> Hybrid Engine & Multi-Plot Insights'
        }

    def _generate_eda_viz_questions(self, df: pd.DataFrame) -> list:
        questions = []
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in numeric_cols[:2]: questions.append(f"Plot the distribution of {col}")
        for col in categorical_cols[:2]:
            if df[col].nunique() < 20: questions.append(f"Show the counts for each category in {col}")
        
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr(numeric_only=True).abs()
                np.fill_diagonal(corr_matrix.values, 0)
                if not corr_matrix.empty:
                    col1, col2 = corr_matrix.unstack().idxmax()
                    questions.append(f"Explore the relationship between {col1} and {col2}")
            except Exception:
                questions.append(f"Explore the relationship between {numeric_cols[0]} and {numeric_cols[1]}")
        return questions

    def _handle_descriptive_question(self, question: str, df: pd.DataFrame, dataset_name: str, classification) -> Dict[str, Any]:
        print("📋 Routing Descriptive Question...")
        q_lower = question.lower()
        if 'columns' in q_lower or 'variables' in q_lower:
            answer = f"The dataset '{dataset_name}' contains {len(df.columns)} columns: {', '.join(df.columns.tolist())}"
            method = 'descriptive_summary'
        elif 'shape' in q_lower or 'size' in q_lower:
            answer = f"The dataset has {df.shape[0]:,} rows and {df.shape[1]} columns."
            method = 'descriptive_summary'
        else:
            print("🧠 Rerouting to InsightAgent for a qualitative answer...")
            answer = self.insight_agent.generate_descriptive_answer(question, df, dataset_name)
            method = f'LLM-Classified ({getattr(classification.intent, "value", "descriptive")}) -> Insight Agent'
        return {
            'question': question, 'answer': answer, 'visualization_html': None,
            'visualization_json': None, 'method': method
        }