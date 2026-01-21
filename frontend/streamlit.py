import streamlit as st
import requests
import os
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

BACKEND = "http://localhost:8000"

st.title("ContextAI – Data Analyst")

# File upload
uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "json", "pdf", "txt"])

if uploaded:
    files = {"file": uploaded}
    res = requests.post(f"{BACKEND}/upload", files=files)
    st.success("File loaded")
    st.write("Suggested Questions:")
    for q in res.json().get("questions", []):
        st.write("-", q)

# Chat
query = st.text_input("Ask a question")

if st.button("Run Analysis"):
    res = requests.post(
        f"{BACKEND}/query",
        json={"message": query}
    )
    data = res.json()

    if data.get("answer"):
        st.subheader("Answer")
        st.write(data["answer"])

    if data.get("analysis", {}).get("visualization_html"):
        st.components.v1.html(
            data["analysis"]["visualization_html"],
            height=500
        )
# ---------------------------
# Generate Report Section
# ---------------------------
# ---------------------------
# Report Section (ADD BELOW)
# ---------------------------
st.divider()
st.subheader("Report")

# Generate report
if st.button("Generate Report"):
    with st.spinner("Generating report..."):
        res = requests.post(
            f"{BACKEND}/query",
            json={"message": "generate report"}
        )

    if res.status_code == 200:
        st.success("Report generated successfully!")
        st.session_state.report_ready = True

# Download report
if st.session_state.report_ready:
    if st.button("Download Report"):
        res = requests.get(f"{BACKEND}/download-report")

        st.write("Status code:", res.status_code)
        st.write("Content-Type:", res.headers.get("content-type"))

        if res.headers.get("content-type") == "application/pdf":
            st.download_button(
                label="Download Report",
                data=res.content,
                file_name="ContextAI_Report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Backend response:")
            st.text(res.text)




