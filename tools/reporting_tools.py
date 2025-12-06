import os
import re
import tempfile
import pandas as pd
from typing import Dict, List
from datetime import datetime

# LangChain import for creating the tool
from langchain_core.tools import tool

# ReportLab imports for PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Plotly imports for embedding visualizations
try:
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ================== THE LANGCHAIN TOOL ==================
# This is the primary function that the "Report Writer" agent will call.

@tool
def generate_comprehensive_report(
    dataset_name: str,
    profile: dict,
    insights: List[Dict],
    questions: List[str],
    understanding: dict,
    dataframe: pd.DataFrame
) -> str:
    """
    Generates a comprehensive PDF report summarizing the dataset profile, initial questions,
    and all the analysis performed. It embeds visualizations from the analysis insights.
    
    Args:
        dataset_name (str): The name of the dataset.
        profile (dict): The detailed profile of the dataset.
        insights (List[Dict]): A list of analysis results, where each result is a dictionary
                               containing the question, answer, and visualization data.
        questions (List[str]): The list of initially generated questions for the dataset.
        understanding (dict): The LLM's understanding of the dataset's domain and purpose.
        dataframe (pd.DataFrame): The cleaned pandas DataFrame for statistical summaries.
        
    Returns:
        str: The absolute file path of the generated PDF report.
    """
    print(f"📄 Tool 'generate_comprehensive_report' starting for dataset: {dataset_name}")

    # Define output path
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles for the report
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=20, textColor=colors.darkblue, alignment=1)
    subtitle_style = ParagraphStyle('CustomSubtitle', parent=styles['Heading2'], fontSize=16, spaceAfter=15, textColor=colors.darkgreen)
    
    story = []

    # --- Build the report story ---

    # 1. Title Page
    story.append(Paragraph("Enhanced Data Analysis Report", title_style))
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(f"<b>Dataset:</b> {dataset_name.replace('_', ' ').title()}", styles['Heading2']))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Domain:</b> {understanding.get('domain', 'General')}", styles['Normal']))
    story.append(PageBreak())

    # 2. Executive Summary & Dataset Overview
    story.append(Paragraph("📋 Executive Summary & Overview", subtitle_style))
    summary_text = f"""
    This report details the analysis of the <b>{dataset_name}</b> dataset, identified as belonging to the <b>{understanding.get('domain', 'General')}</b> domain.
    The dataset contains <b>{dataframe.shape[0]:,}</b> records and <b>{dataframe.shape[1]}</b> features.
    This document outlines the dataset's structure, key questions explored, and the insights derived from the analysis.
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 15))

    # 3. Initially Generated Questions
    if questions:
        story.append(Paragraph("❓ Initial Analysis Questions", subtitle_style))
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"{i}. {q}", styles['Normal']))
        story.append(PageBreak())

    # 4. User Analysis Sessions (Insights)
    if insights:
        story.append(Paragraph("🔍 Analysis Insights", subtitle_style))
        for i, session in enumerate(insights, 1):
            question = session.get('question', 'N/A')
            answer = session.get('answer', 'No answer recorded.')
            
            story.append(Paragraph(f"<b>Insight {i}: {question}</b>", styles['Heading3']))
            story.append(Paragraph(format_answer_for_pdf(answer), styles['Normal']))
            
            # Embed visualization if it exists and Plotly is available
            viz_json = session.get('visualization_json')
            if viz_json and PLOTLY_AVAILABLE:
                try:
                    fig = pio.from_json(viz_json)
                    # Use a temporary file to save the image for embedding
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        img_path = tmp_file.name
                        pio.write_image(fig, img_path, format="png", width=800, height=500, scale=2)
                        story.append(Spacer(1, 10))
                        story.append(Image(img_path, width=450, height=281.25)) # 16:10 aspect ratio
                        os.remove(img_path) # Clean up the temp file
                except Exception as e:
                    print(f"Warning: Could not embed visualization for '{question}': {e}")
                    story.append(Paragraph("<i>[Visualization could not be embedded in PDF.]</i>", styles['Italic']))

            story.append(Spacer(1, 20))
            if i % 2 == 0 and i < len(insights):
                story.append(PageBreak())

    # Build the PDF
    try:
        doc.build(story)
        abs_path = os.path.abspath(output_path)
        print(f"✅ Report generation successful. Saved to: {abs_path}")
        return abs_path
    except Exception as e:
        error_message = f"Error: Could not build PDF report. {e}"
        print(f"❌ {error_message}")
        return error_message


# ========== INTERNAL HELPER FUNCTIONS ==========
# These functions are used by the main tool to format the PDF content.

def format_answer_for_pdf(answer_text: str) -> str:
    """Cleans and formats the LLM's text answer for better PDF readability."""
    if not isinstance(answer_text, str):
        return ""
    # Remove special characters/emojis used for console display
    clean_text = re.sub(r'[📊📈📉💡🔍⚡🧠🎯📋⚙️🎲•❓✅❌✔️]+', '', answer_text)
    # Convert markdown bold to reportlab bold
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_text)
    # Replace newlines with HTML line breaks
    clean_text = clean_text.replace('\n', '<br/>')
    # Remove excessive line breaks
    clean_text = re.sub(r'(<br/>\s*){3,}', '<br/><br/>', clean_text).strip()
    return clean_text