import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mentor_chat import ask_mentor
from memory_db import init_db, add_note, get_memory_context

app = FastAPI(title="The Sanctuary")
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str

class NoteRequest(BaseModel):
    category: str
    content: str

def route_room(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ["sql", "schema", "pipeline", "deploy", "architecture", "api", "docker"]):
        return "command_center"
    if any(k in p for k in ["metric", "eval", "chart", "pandas", "model", "r ", "regression", "accuracy"]):
        return "analytics_studio"
    if any(k in p for k in ["project", "resume", "memory", "history", "note", "github", "portfolio"]):
        return "library_archive"
    return "habitat_lounge"

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.get("/api/notes")
def fetch_notes():
    notes_block, _ = get_memory_context(limit_messages=4)
    return {"notes": notes_block}

@app.get("/api/benchmarks") 
def fetch_benchmarks():
    bench_path = Path("benchmark_comparison.json")
    if not bench_path.exists():
        return {"items": [], "error": "No benchmark_comparison.json found."}
    try:
        raw = json.loads(bench_path.read_text(encoding="utf-8"))
        # Normalize whether benchmark_comparison.json is a list or dict
        entries = raw if isinstance(raw, list) else raw.get("results", [])
        cleaned = []
        for item in entries[:6]:
            cleaned.append({
                "id": item.get("id", "eval"),
                "category": item.get("category", "general"),
                "prompt": item.get("prompt", ""),
                "base": item.get("base_model_before", "N/A"),
                "tuned": item.get("fine_tuned_after", item.get("tuned_model_after", "N/A")),
            })
        return {"items": cleaned}
    except Exception as e:
        return {"items": [], "error": str(e)}
    
    bench_path = Path("benchmark_comparison.json")
    if not bench_path.exists():
        return {"summary": "No benchmark_comparison.json file found in root yet."}
    try:
        data = json.loads(bench_path.read_text(encoding="utf-8"))
        return {"summary": json.dumps(data, indent=2)[:900]}
    except Exception as e:
        return {"summary": f"Could not parse benchmark_comparison.json: {e}"}

@app.post("/api/notes")
def create_note(req: NoteRequest):
    add_note(req.category, req.content)
    return {"status": "saved"}

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    room = route_room(req.message)
    reply = ask_mentor(req.message)
    return {"room": room, "reply": reply}