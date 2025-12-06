import json
from typing import Dict
from langchain_core.tools import tool
from gemma_llm import GemmaLLM

@tool
def critique_analysis(user_question: str, analysis_response: Dict) -> Dict:
    """
    Critiques a given data analysis response based on the original user question.
    It uses an LLM to score the analysis for relevance, accuracy, and clarity,
    returning a score and a constructive critique. This tool is used to validate
    analytical outputs before they are finalized.

    Args:
        user_question (str): The user's original question.
        analysis_response (Dict): The dictionary containing the analysis output.

    Returns:
        dict: A dictionary with 'score' (int) and 'critique' (str).
    """
    print("---TOOL: Critic Tool---")
    
    # Extract the text answer for the prompt
    answer_text = analysis_response.get('answer', 'No answer provided.')

    prompt = f"""
    You are an expert AI Data Analyst Critic. Evaluate the following analysis based on a user's question.

    **User's Question:** "{user_question}"
    **AI's Generated Answer:** "{answer_text}"

    **Critique Guidelines:**
    - Score **Relevance**: Does it directly answer the question?
    - Score **Accuracy**: Is the information correct and well-founded?
    - Score **Clarity**: Is the answer easy to understand?

    **Output Format (JSON only):**
    Provide a JSON object with two keys:
    1.  `"score"`: An integer from 1 to 10 (a score below 7 requires a rethink).
    2.  `"critique"`: A brief, constructive critique explaining the score.

    **Your Evaluation (JSON only):**
    """
    try:
        critic_llm = GemmaLLM(model_name="google/gemma-2-9b-it", temperature=0.1)
        critique_text = critic_llm(prompt)
        cleaned_text = critique_text.strip().replace('```json', '').replace('```', '').strip()
        critique_json = json.loads(cleaned_text)
        print(f"---Critic Tool: Score {critique_json.get('score', 0)}/10---")
        return {
            "critique": critique_json.get('critique'),
            "score": critique_json.get('score', 0)
        }
    except Exception as e:
        print(f"---Critic Tool: Error -> {e}---")
        return {"critique": "Failed to generate a critique.", "score": 0}