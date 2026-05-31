import streamlit as st
import tempfile
import os
import sys
import base64
import streamlit.components.v1 as components
import requests

BACKEND = "https://contextai-backend-7o1w.onrender.com"

st.set_page_config(page_title="ContextAI – Agentic Data Analyst", layout="wide")
st.title("ContextAI – Agentic Data Analyst")

# -----------------------------
# File Upload
# -----------------------------
uploaded = st.file_uploader(
    "📂 Upload dataset",
    type=["csv", "xlsx", "json", "pdf", "txt"]
)

if uploaded:
    with st.spinner("Uploading dataset..."):
        files = {"file": uploaded}
        res = requests.post(f"{BACKEND}/upload", files=files)

    if res.status_code == 200:
        data = res.json()
        st.success("✅ Dataset loaded")

        if data.get("questions"):
            st.subheader("💡 Suggested Questions")
            for q in data["questions"]:
                st.write("•", q)
    else:
        st.error("❌ Failed to upload dataset")

# -----------------------------
# Chat / Analysis
# -----------------------------
st.subheader("💬 Ask a question")
query = st.text_input("Enter your question")

if st.button("Run Analysis"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing..."):
            res = requests.post(
                f"{BACKEND}/query",
                json={"message": query}
            )

        if res.status_code == 200:
            data = res.json()

            if data.get("answer"):
                st.subheader("🧠 Answer")
                st.write(data["answer"])

            analysis = data.get("analysis")
            if analysis and analysis.get("visualization_html"):
                st.subheader("📊 Visualization")
                components.html(
                    analysis["visualization_html"],
                    height=500
                )
        else:
            st.error("❌ Analysis failed")

# -----------------------------
# Report Generation
# -----------------------------
st.subheader("📄 Report")

if st.button("Generate Report"):
    with st.spinner("Generating report..."):
        res = requests.post(
            f"{BACKEND}/query",
            json={"message": "Generate a comprehensive report"}
        )

    if res.status_code == 200:
        st.success("✅ Report generated")
        st.session_state["report_ready"] = True  # ✅ flag it
    else:
        st.error("❌ Report generation failed")

# -----------------------------
# Report Preview + Download
# -----------------------------
# ✅ Only fetch report when user has generated one
if st.session_state.get("report_ready"):
    st.subheader("👀 Report Preview")

    with st.spinner("Loading report..."):
        preview = requests.get(f"{BACKEND}/download-report")  # ✅ moved inside guard

    if preview.status_code == 200:
        pdf_bytes = preview.content
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        components.html(
            f"""
            <iframe
                src="data:application/pdf;base64,{pdf_base64}"
                width="100%"
                height="800px"
                style="border:none;"
            ></iframe>
            """,
            height=820
        )

        st.download_button(
            label="⬇️ Download Report",
            data=pdf_bytes,
            file_name="contextai_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("⚠️ Report not available yet. Generate one first.")