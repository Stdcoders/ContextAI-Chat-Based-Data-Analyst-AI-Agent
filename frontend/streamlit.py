import streamlit as st
import tempfile
import os

from agentworkflow import AgentState, run_contextai
import streamlit.components.v1 as components

st.set_page_config(page_title="ContextAI – Agentic Data Analyst", layout="wide")
st.title("ContextAI – Agentic Data Analyst")

# -----------------------------
# Initialize Agent State
# -----------------------------
if "state" not in st.session_state:
    st.session_state.state = AgentState(
        dataset_name=None,
        file_path=None,
        dataframe=None,
        df_profile=None,
        understanding=None,
        questions=[],
        analysis_history=[],
        user_request="",
        is_cleaned=False,
        chat_history=[],
        report_path=None
    )

STATE = st.session_state.state

# -----------------------------
# File Upload
# -----------------------------
uploaded = st.file_uploader(
    "📂 Upload dataset",
    type=["csv", "xlsx", "json", "pdf", "txt"]
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    with st.spinner("Loading dataset and generating questions..."):
        STATE, response = run_contextai(
            state=STATE,
            user_input="",
            file_path=tmp_path
        )

    st.session_state.state = STATE

    st.success("✅ Dataset loaded")

    if response.get("questions"):
        st.subheader("💡 Suggested Questions")
        for q in response["questions"]:
            st.write("•", q)

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
            STATE, response = run_contextai(
                state=STATE,
                user_input=query
            )

        st.session_state.state = STATE

        if response.get("answer"):
            st.subheader("🧠 Answer")
            st.write(response["answer"])

        analysis = response.get("analysis")
        if analysis and analysis.get("visualization_html"):
            st.subheader("📊 Visualization")
            components.html(
                analysis["visualization_html"],
                height=500
            )

# -----------------------------
# Report Generation
# -----------------------------
st.subheader("📄 Report")

if st.button("Generate Report"):
    with st.spinner("Generating report..."):
        STATE, response = run_contextai(
            state=STATE,
            user_input="Generate a comprehensive report"
        )

    st.session_state.state = STATE

    if STATE.get("report_path"):
        st.success("✅ Report generated successfully")
    else:
        st.error("❌ Report generation failed")

# -----------------------------
# Report Preview
# -----------------------------
if STATE.get("report_path") and os.path.exists(STATE["report_path"]):
    st.subheader("👀 Report Preview")

    with open(STATE["report_path"], "rb") as f:
        pdf_bytes = f.read()

    components.html(
        f"""
        <iframe
            src="data:application/pdf;base64,{pdf_bytes.hex()}"
            width="100%"
            height="800px"
            style="border:none;"
        ></iframe>
        """,
        height=820
    )

    # -----------------------------
    # Report Download
    # -----------------------------
    st.download_button(
        label="⬇️ Download Report",
        data=pdf_bytes,
        file_name=os.path.basename(STATE["report_path"]),
        mime="application/pdf"
    )
