import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("Missing DATABASE_URL in .env")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(db_url, pool_pre_ping=True)

def init_db():
    """No-op compatibility helper so mentor_chat.py runs unchanged."""
    pass

def add_note(category: str, content: str):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO mentor_sanctuary.project_notes (category, content) VALUES (:cat, :txt)"),
            {"cat": category, "txt": content},
        )

def get_memory_context(limit_messages: int = 4) -> tuple[str, list[dict]]:
    with engine.connect() as conn:
        notes = conn.execute(
            text("SELECT category, content FROM mentor_sanctuary.project_notes ORDER BY id DESC LIMIT 10")
        ).fetchall()
        messages = conn.execute(
            text("SELECT role, content FROM mentor_sanctuary.chat_history ORDER BY id DESC LIMIT :lim"),
            {"lim": limit_messages},
        ).fetchall()

    notes_block = "\n".join(f"- [{cat}] {txt}" for cat, txt in notes) or "No saved notes yet."
    history = [{"role": r, "content": c} for r, c in reversed(messages)]
    return notes_block, history

def save_turn(role: str, content: str):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO mentor_sanctuary.chat_history (role, content) VALUES (:r, :c)"),
            {"r": role, "c": content},
        )

if __name__ == "__main__":
    projects = [
        ("Portfolio - Python", "WildTrack: Live GBIF wildlife tracking app with Supabase/PostgreSQL, Plotly 3D globe & treemaps, and DeepSeek field reports."),
        ("Portfolio - SQL", "K-12 District ROI Pipeline: 3NF PostgreSQL database in Supabase linking NCES F-33 finance and EDFacts scores across 7,726 districts."),
        ("Portfolio - R Stats", "NFL Crowd Bias: Econometric regression in R testing discretionary vs. procedural referee penalties using 2018-2022 nflverse data."),
        ("Portfolio - R Munging", "AI Threat to Jobs: Joined AI career threat index with SOC occupational data to model risk across income tiers and sectors."),
        ("Portfolio - Applied AI", "The Sanctuary (mentor-llm): Custom Qwen3-4B-Instruct fine-tuned with Unsloth SFT + DPO and deployed locally via GGUF/Ollama."),
        ("Research / Product", "Track & Field Biomechanics & EDU Hub: Adapting computer-vision motion tracking to sprint kinematics and educational tools."),
    ]
    for category, content in projects:
        add_note(category, content)
    print("Supabase memory seeded! Preview:\n", get_memory_context()[0]) #to make sure it has context of my previous projects