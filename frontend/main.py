from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os, shutil

from agentworkflow import AgentState, run_contextai
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(title="ContextAI Backend")

STATE = AgentState(
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

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    message: str

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    global STATE
    STATE, response = run_contextai(
        state=STATE,
        user_input="",
        file_path=path
    )

    return {
        "status": "loaded",
        "questions": response["questions"]
    }

@app.post("/query")
def query(req: QueryRequest):
    global STATE
    STATE, response = run_contextai(
        state=STATE,
        user_input=req.message
    )

    if "report_path" in response:
        STATE.last_report_path = response["report_path"]
        print("Stored report path:", STATE.last_report_path)


    return response

@app.get("/download-report")
def download_report():
    global STATE

    report_path = getattr(STATE, "last_report_path", None)

    print("DOWNLOAD ENDPOINT HIT")
    print("Stored report path:", report_path)

    if not report_path:
        print("❌ No report path in STATE")
        return {"error": "Report path missing"}

    if not os.path.exists(report_path):
        print("❌ Report file does not exist on disk")
        return {"error": "Report file not found"}

    print("✅ Returning report file")

    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=os.path.basename(report_path)
    )



