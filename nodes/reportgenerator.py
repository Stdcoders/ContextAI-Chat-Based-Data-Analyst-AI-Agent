from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
import os
import pandas as pd
import plotly.io as pio
from utils.state import STATE

def generate_report(state: STATE, dataset_name: str, output_path="data_insights_report.pdf") -> str:
    """
    Generate PDF report using profile, insights, and questions from WorkflowState.
    Updates state with report path.
    """
    profile = state.profiles.get(dataset_name, {})
    insights = state.insights.get(dataset_name, [])

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"📊 Report for Dataset: {dataset_name}", styles["Title"]))
    story.append(Spacer(1, 24))

    # Profile Summary
    story.append(Paragraph("Profile Summary", styles["Heading2"]))
    if "columns" in profile and isinstance(profile["columns"], list):
        df_profile = pd.DataFrame(profile["columns"])
        table_data = [df_profile.columns.tolist()] + df_profile.values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("⚠️ No profile summary available.", styles["Normal"]))
    story.append(Spacer(1, 24))

    # Insights
    story.append(Paragraph("Insights", styles["Heading2"]))
    if insights:
        for i, ins in enumerate(insights, 1):
            story.append(Paragraph(f"Q{i}: {ins['question']}", styles["Heading3"]))
            story.append(Paragraph(ins["answer"], styles["Normal"]))
            story.append(Spacer(1, 12))

            if ins.get("visualization_html"):
                png_path = f"{dataset_name}_insight_{i}.png"
                try:
                    fig = pio.from_json(ins["visualization_html"])
                    pio.write_image(fig, png_path, format="png", width=600, height=400, scale=2)
                    story.append(Image(png_path, width=400, height=300))
                except Exception:
                    story.append(Paragraph("⚠️ Visualization could not be embedded in PDF.", styles["Normal"]))
            story.append(Spacer(1, 24))
    else:
        story.append(Paragraph("⚠️ No insights available yet.", styles["Normal"]))

    story.append(Spacer(1, 48))
    doc.build(story)

    abs_path = os.path.abspath(output_path)
    print(f"✅ Report generated at: {abs_path}")

    # Update global state
    if not hasattr(state, "reports"):
        state.reports = {}  # add dynamically if missing
    state.reports[dataset_name] = abs_path

    return abs_path
