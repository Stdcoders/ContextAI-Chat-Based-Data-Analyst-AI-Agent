import pandas as pd
from typing import TypedDict, List, Dict, Optional, Tuple
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import os
import re

# Import your refactored tool functions
from tools.data_ingestion_tools import ingest_data
from tools.data_cleaning_tools import clean_and_analyze_data
from tools.data_exploration_tools import understand_data, generate_questions
from tools.analysis_tools import analyze_user_question
from tools.reporting_tools import generate_comprehensive_report
import os
import re
from datetime import datetime

def save_visualization(html_content: str, question: str):
    """Saves the plot's HTML content to a file."""
    try:
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)

        # Create a safe filename from the question
        safe_filename = re.sub(r'[^a-zA-Z0-9\s]', '', question).strip()
        safe_filename = re.sub(r'\s+', '_', safe_filename)[:50] # Truncate long names

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{safe_filename}_{timestamp}.html")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📁 Graph saved to: {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")

## 1. Define the State for the Graph
# This is the shared memory for all agents.
class AgentState(TypedDict):
    dataset_name: Optional[str]
    file_path: Optional[str]
    dataframe: Optional[pd.DataFrame]
    df_profile: Optional[Dict]
    understanding: Optional[Dict]
    questions: List[str]
    analysis_history: List[Dict]
    user_request: str
    is_cleaned: bool
    chat_history: List[Tuple[str, str]]

## 2. Define the Agent Nodes
# Each node represents a specialist agent's turn.

def data_steward_node(state: AgentState):
    """Agent node for ingesting data."""
    print("---AGENT: Data Steward---")
    file_path = state['file_path']
    dataset_name = state['dataset_name']
    
    ext = os.path.splitext(file_path)[1].lower()
    file_type_map = {
        '.csv': 'csv', '.xlsx': 'excel', '.xls': 'excel',
        '.json': 'json', '.pdf': 'pdf', '.txt': 'text'
    }
    file_type = file_type_map[ext]

    result = ingest_data.invoke({
        "dataset_name": dataset_name,
        "file_path": file_path,
        "file_type": file_type
    })
    
    print("---Data Steward: Ingestion complete.---")
    return {
        "dataframe": result["dataframe"],
        "df_profile": result["profile"]
    }

# In multi_agenticworkflow.py

def data_janitor_node(state: AgentState):
    """Agent node for cleaning data."""
    print("---AGENT: Data Janitor---")
    df = state['dataframe']
    dataset_name = state['dataset_name']

    # Call the cleaning tool
    cleaning_result = clean_and_analyze_data.invoke({
        "df": df,
        "dataset_name": dataset_name
    })

    # Extract cleaned dataframe and preview
    cleaned_dataframe = cleaning_result.get("cleaned_dataframe", df)
    sample_preview = cleaning_result.get("sample_preview", [])

    print("\n✅ --- Data Janitor: Cleaning complete. ---")
    print(f"📊 Cleaned dataset shape: {cleaned_dataframe.shape}")

    # Return updated state
    return {
        "dataframe": cleaned_dataframe,  # cleaned DataFrame object
        "is_cleaned": True
    }

def data_explorer_node(state: AgentState):
    """Agent node for understanding and generating questions."""
    print("---AGENT: Data Explorer---")
    profile = state['df_profile']
    dataset_name = state['dataset_name']

    understanding = understand_data.invoke({"df_profile": profile})
    questions = generate_questions.invoke({
        "df_profile": profile,
        "understanding": understanding,
        "num_questions": 8
    })
    
    print("---Data Explorer: Initial analysis complete.---")
    return {
        "understanding": understanding,
        "questions": questions
    }

# You'll need to import the new class at the top of multi_agenticworkflow.py
# from nodes.unified_analysis_engine import UnifiedAnalysisEngine

# Replace your old node with this clean version
def insight_analyst_node(state: AgentState):
    """Agent node that uses the unified analysis tool."""
    print("---AGENT: Insight Analyst---")
    df = state['dataframe']
    question = state['user_request']
    dataset_name = state['dataset_name']
    chat_history = state['chat_history'] # Get the history from the state

    # Pass the history to your tool/agent
    analysis_result = analyze_user_question.invoke({
        "question": question,
        "df": df,
        "dataset_name": dataset_name,
        "chat_history": chat_history # Add this key
    })
    
    
    # Append the result to the history
    history = state.get('analysis_history', [])
    history.append(analysis_result)
    
    print("---Insight Analyst: Question answered.---")
    print("\n---ANALYSIS RESULT---")
    print(analysis_result.get('answer'))
    
    # Handle and save the visualization if it exists
    viz_html = analysis_result.get('visualization_html')
    if viz_html:
        save_visualization(viz_html, analysis_result.get('question', 'untitled_plot'))
        
    return {"analysis_history": history}

def report_writer_node(state: AgentState):
    """Agent node for generating the final PDF report."""
    print("---AGENT: Report Writer---")
    
    report_path = generate_comprehensive_report.invoke({
        "dataset_name": state['dataset_name'],
        "profile": state['df_profile'],
        "insights": state['analysis_history'],
        "questions": state['questions'],
        "understanding": state['understanding'],
        "dataframe": state['dataframe'] # Pass dataframe for stats
    })
    
    print(f"---Report Writer: Report generated at {report_path}---")
    return {}

## 3. Define the Router
# This function decides which agent to call next.

# In multi_agent_workflow.py

# In multi_agenticworkflow.py
# In multi_agenticworkflow.py

# THIS IS THE CORRECTED CODE TO USE

# In multi_agenticworkflow.py

def planner_router(state: AgentState) -> str: # <-- Change return type hint to str
    """
    The central planner that determines the next step by returning the name of the next node.
    """
    print("---ROUTER: Planning next action---")
    user_request = state['user_request'].lower()

    if "report" in user_request:
        print("---ROUTER: Decision -> report_writer---")
        return "report_writer"
    
    if state.get('dataframe') is None:
        print("---ROUTER: Decision -> data_steward---")
        return "data_steward"
        
    if not state.get('is_cleaned', False):
        print("---ROUTER: Decision -> data_janitor---")
        return "data_janitor"
        
    if not state.get('questions'):
        print("---ROUTER: Decision -> data_explorer---")
        return "data_explorer"
    
    # If data is loaded and explored, any new prompt is a question for the analyst
    print("---ROUTER: Decision -> insight_analyst---")
    return "insight_analyst"

## 4. Assemble the Graph

workflow = StateGraph(AgentState)

# Add nodes for each agent
workflow.add_node("data_steward", data_steward_node)
workflow.add_node("data_janitor", data_janitor_node)
workflow.add_node("data_explorer", data_explorer_node)
workflow.add_node("insight_analyst", insight_analyst_node)
workflow.add_node("report_writer", report_writer_node)
# workflow.add_node("planner_router", planner_router)

# The entry point is the router
# workflow.set_entry_point("planner_router")
workflow.set_conditional_entry_point(
    planner_router,
    {
        "data_steward": "data_steward",
        "data_janitor": "data_janitor",
        "data_explorer": "data_explorer",
        "insight_analyst": "insight_analyst",
        "report_writer": "report_writer",
        "__end__": END
    }
)

# Define the workflow path after each agent completes its task
workflow.add_edge("data_steward", "data_janitor")
workflow.add_edge("data_janitor", "data_explorer")
workflow.add_edge("data_explorer", END) # Stop and wait for user question
workflow.add_edge("insight_analyst", END) # Stop and wait for next command
workflow.add_edge("report_writer", END) # Stop after generating report

# Compile the graph into a runnable application
app = workflow.compile()


## 5. Main Execution Loop

# ============================
# FASTAPI-COMPATIBLE WRAPPER
# ============================

def run_contextai(
    state: AgentState,
    user_input: str,
    file_path: str | None = None
) -> tuple[AgentState, dict]:

    # Handle file loading
    if file_path:
        state["file_path"] = file_path
        state["dataset_name"] = os.path.splitext(os.path.basename(file_path))[0]
        state["user_request"] = f"Load the file {file_path}"
    else:
        state["user_request"] = user_input

    # Invoke LangGraph
    result = app.invoke(state)

    # Update chat history if analysis happened
    if result.get("analysis_history"):
        latest = result["analysis_history"][-1]
        state["chat_history"].append(
            (user_input, latest.get("answer", ""))
        )

    # Update full state
    state.update(result)

    # ---------------- RESPONSE PAYLOAD ----------------
    response = {
        "answer": None,
        "questions": state.get("questions", []),
        "analysis": None,
        "report_generated": False
    }

    # Latest analysis
    if state.get("analysis_history"):
        response["analysis"] = state["analysis_history"][-1]
        response["answer"] = response["analysis"].get("answer")

        # 🔥 CAPTURE REPORT PATH FROM ANALYSIS
        if "report_path" in response["analysis"]:
            response["report_path"] = response["analysis"]["report_path"]

    # 🔥 FALLBACK: check top-level result (LangGraph tool output)
    if "report_path" in result:
        response["report_path"] = result["report_path"]

    # Flag report intent
    if "report" in user_input.lower():
        response["report_generated"] = True

    return state, response



if __name__ == "__main__":
    print("ContextAI- Your own data analyst agent!")
    print("Type 'load <path_to_file>' to begin, 'report' to generate a report, or a question to analyze.")
    
    # Initialize the state
    state = AgentState(
        dataset_name=None,
        file_path=None,
        dataframe=None,
        df_profile=None,
        understanding=None,
        questions=[],
        analysis_history=[],
        user_request="",
        is_cleaned=False,
        chat_history=[]
    )
    
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # Check for 'load' command
        if user_input.lower().startswith('load '):
            path = user_input[5:].strip().strip("'\"")
            if os.path.exists(path):
                state['file_path'] = path
                state['dataset_name'] = os.path.splitext(os.path.basename(path))[0]
                state['user_request'] = f"Load the file {path}"
            else:
                print(f"File not found: {path}")
                continue
        else:
            state['user_request'] = user_input
            
        # Invoke the agent graph
        result = app.invoke(state)
        if result.get('analysis_history'):
            latest_answer = result['analysis_history'][-1].get('answer', '')
            state['chat_history'].append((user_input, latest_answer))
        # Update the state with the results from the run
        state.update(result)
        
        # After initial load and exploration, prompt for questions
        if state.get('questions'):
            print("\n---Initial exploration complete. You can now ask questions about the data or request a 'report'.---")
            print("Suggested Questions:")
            for i, q in enumerate(state['questions'], 1):
                print(f"{i}. {q}")