import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import re

# Import your LLM wrappers
try:
    from deepseek_llm import DeepSeekLLM
    from gemma_llm import GemmaLLM
    from groq_llm import GroqLLM
except ImportError:
    print("FATAL ERROR: Could not import LLM modules.")
    # Define minimal fallbacks
    class BaseLLM:
        def __init__(self, **kwargs): pass
        def is_available(self): return False
        def __call__(self, prompt): return ""
    DeepSeekLLM = GemmaLLM = GroqLLM = BaseLLM

class HybridCalculationEngine:
    """
    LLM-powered engine that generates and executes multi-step pandas/scipy/statsmodels
    code for maximum accuracy and flexibility in complex analytical tasks.
    """
    def __init__(self):
        self.code_llm = GroqLLM(model_name="openai/gpt-oss-120b", temperature=0.0, max_tokens=2000)
        self.context_llm = GemmaLLM(model_name="google/gemma-2-9b-it", temperature=0.2, max_tokens=1500)
        print("🔥 Initializing Advanced Multi-Step Hybrid Calculation Engine...")
        print(f"   ⚡ Code Generation (Groq): {'✅ Available' if self.code_llm.is_available() else '❌ Unavailable'}")
        print(f"   🧠 Contextual Explanation (Gemma): {'✅ Available' if self.context_llm.is_available() else '❌ Unavailable'}")

    def analyze_with_calculations(self, question: str, df: pd.DataFrame) -> str:
        if not self.code_llm.is_available():
            return "Calculation engine is not available."

        # Step 1: Generate a Python script
        code_gen_prompt = self._create_code_generation_prompt(question, df)
        generated_script = self.code_llm(code_gen_prompt)

        # Ensure we are working with a string
        if not isinstance(generated_script, str):
             if hasattr(generated_script, 'content'):
                 generated_script = generated_script.content or ""
             else:
                 generated_script = str(generated_script)

        cleaned_script = self._clean_generated_code(generated_script)
        if not cleaned_script:
            return "Could not generate a valid Python script for this complex analytical question."

        # Step 2: Execute the script
        calculation_result = self._execute_script_and_format_result(cleaned_script, df)

        # Step 3: Get a contextual explanation
        explanation = self._get_contextual_explanation(question, calculation_result)

        # Step 4: Combine and return
        return self._combine_results(calculation_result, explanation, cleaned_script)

    def _create_code_generation_prompt(self, question: str, df: pd.DataFrame) -> str:
        col_details = "\n".join([f"- '{col}' (type: {dtype})" for col, dtype in df.dtypes.items()])
        return (
            f"You are a senior data scientist. Your task is to write a Python script to answer the user's question using a pandas DataFrame named `df`.\n\n"
            f"USER QUESTION: \"{question}\"\n\n"
            f"DATAFRAME COLUMNS:\n{col_details}\n\n"
            "AVAILABLE LIBRARIES: pandas as pd, numpy as np, scipy.stats as stats, statsmodels.api as sm\n\n"
            "RULES:\n"
            "1. Write a Python script (it can be multi-line).\n"
            "2. The script must use the DataFrame named `df`.\n"
            "3. The final answer (a DataFrame, Series, or string) MUST be stored in a variable called `result`.\n"
            "4. Do not include any sample data creation (e.g., `pd.DataFrame(...)`).\n"
            "5. The output must be ONLY the Python code, with no explanations or markdown.\n\n"
            "EXAMPLE for a complex question:\n"
            "Question: 'What is the correlation between charges and bmi for smokers only?'\n"
            "smokers_df = df[df['smoker'] == 'yes']\n"
            "result = smokers_df[['charges', 'bmi']].corr()\n\n"
            "PYTHON SCRIPT:"
        )

    def _clean_generated_code(self, code: str) -> str:
        return str(code).strip().replace("`python", "").replace("`", "")

    def _execute_script_and_format_result(self, script: str, df: pd.DataFrame) -> str:
        """
        Executes a multi-line script using exec() and captures the 'result' variable.
        """
        try:
            print(f"Executing script on full DataFrame:\n---\n{script}\n---")
            # Prepare the execution environment
            local_scope = {"df": df, "pd": pd, "np": np}
            # Import stats libraries for advanced analysis
            exec("import scipy.stats as stats", local_scope)
            exec("import statsmodels.api as sm", local_scope)

            # Execute the script
            exec(script, local_scope)

            # Retrieve the result
            result = local_scope.get('result', None)
            if result is None:
                return "**Error Executing Code:**\nThe script ran successfully, but it did not store a value in the `result` variable."

            # Format the result
            if isinstance(result, (pd.Series, pd.DataFrame)):
                return result.to_markdown()
            else:
                return f"**Result:**\n{str(result)}"
        except Exception as e:
            return f"**Error Executing Code:**\n```python\n{script}\n```\n\n**Reason:** {e}"

    def _get_contextual_explanation(self, question: str, result: str) -> Optional[str]:
        if not self.context_llm.is_available() or "Error" in result:
            return None
        prompt = (
            f"The following calculation was performed to answer a user's question. Please explain what the result means in a clear, business-oriented context.\n\n"
            f"ORIGINAL QUESTION: {question}\n\n"
            f"CALCULATION RESULT:\n{result}\n\n"
            "EXPLANATION (do NOT repeat the numbers; focus on the meaning and implications):"
        )
        return self.context_llm(prompt)

    def _combine_results(self, calculation: str, explanation: Optional[str], code: str) -> str:
        response = f"## ⚡ **Calculation Result**\n\n{calculation}"
        if explanation:
            response += f"\n\n---\n\n## 🧠 **Analytical Insight**\n\n{explanation}"
        response += f"\n\n---\n*Analysis generated by executing the following script:*\n```python\n{code}\n```"
        return response