import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env file.")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

MENTOR_SYSTEM_PROMPT = """You are a seasoned Senior Data Scientist and Applied AI Engineer mentoring a junior data scientist.
Your style:
- Direct, constructive, slightly witty, and concise (zero corporate fluff or word salad).
- Anti-Yes-Man: Never blindly validate weak ideas. Evaluate code, architecture, and product ideas against the Engineering Big 3:
  1. Simplicity (Maintainability—avoid over-engineering when clean SQL/Pandas works)
  2. Reliability (Data Integrity—handle NULLs, schemas, and edge cases cleanly)
  3. Impact (User/Business Value—solve real problems in clean, scannable UIs)
- Context-aware: You know the user builds with Python, SQL/PostgreSQL (Supabase), R, Pandas, Streamlit, DeepSeek/OpenAI SDKs, and Unsloth, across projects like WildTrack (GBIF biodiversity explorer), a K-12 District ROI 3NF database, an NFL crowd bias econometric study, and sports biomechanics."""

SCENARIOS = [
    # --- Technical & Architecture Trade-Offs ---
    "I'm deciding between fine-tuning a 4B model with QLoRA versus using RAG with a hosted API for a domain-specific tool. How should I weigh the trade-offs?",
    "Should I start with a multi-agent framework for my MVP, or build a single deterministic pipeline first?",
    "When evaluating my fine-tuned model, is training loss enough, or how should I structure a proper before-and-after benchmark?",
    "My dataset has a 92/8 class imbalance. Should I apply SMOTE right away, or start with class weights and precision-recall thresholds?",
    "I'm joining external datasets where keys have formatting drift and privacy-suppressed ranges. How should I design schema validation checks before modeling?",
    "Should I stick with a strict 3NF schema or build denormalized analytical views to power a live dashboard?",
    "What quantization format (like Q4_K_M vs. Q8_0) gives the best balance of speed and reasoning quality for local GGUF inference?",
    "How should I split responsibilities between fine-tuned model weights and external database memory for a local personal assistant?",
    "I have a working statistical model, and I want to turn it into an interactive tool that decision-makers actually enjoy using. What should I prioritize?",

    # --- Career Strategy, Recruiting & Portfolio Positioning ---
    "I'm targeting Data Scientist, Data Analyst, and Applied AI internships/roles. Should I maintain one general resume or tailor separate versions for each track?",
    "How do I frame academic data projects—like an econometric sports analysis or a K-12 school finance SQL database—so hiring managers see real product value instead of homework?",
    "How do I talk openly about model limitations and high sample variance in a technical interview without underselling my work?",
    "What makes an Applied AI portfolio project actually stand out to technical recruiters versus looking like a standard API wrapper?",
    "When applying to fast-moving AI startups, how do I demonstrate that I can ship end-to-end systems rather than just run notebooks?",
    "How should I structure a 60-second walkthrough of a full-stack data project (ETL pipeline, database schema, and LLM app) during a behavioral or hiring manager screen?",
    "I have projects across wildlife conservation, education finance, labor economics, and sports biomechanics. Should I niche down on LinkedIn or pitch cross-domain adaptability?",
    "In a live SQL or Python technical interview, what habits immediately separate a strong junior candidate from someone who just memorizes syntax?",
    "How do I evaluate whether a Data Analyst or Data Science internship will give me real engineering ownership versus stuck doing manual spreadsheet cleanup?",
]


def generate_batch(scenario: str) -> dict | None:
    prompt = f"""Given this user message to their AI mentor:
"{scenario}"

Generate a JSON object with 3 fields:
1. "sft_response": An ideal response following the Senior Data Scientist mentor persona (concise, scannable bullets or short paragraph, references Simplicity/Reliability/Impact where relevant, pushes back constructively if the idea is flawed).
2. "dpo_chosen": A sharp, anti-yes-man mentor critique that directly flags trade-offs and gives a cleaner engineering path.
3. "dpo_rejected": A stereotypical "Yes-Man" AI response that blindly praises the user ("That's a fantastic idea! Here are 8 comprehensive ways...") with bloated advice.

Return ONLY valid JSON with keys "sft_response", "dpo_chosen", and "dpo_rejected"."""

    res = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": MENTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return json.loads(res.choices[0].message.content)


def build_datasets(repeats_per_scenario: int = 3):
    sft_count, dpo_count = 0, 0
    with open("train_sft.jsonl", "a", encoding="utf-8") as sft_f, open(
        "train_dpo.jsonl", "a", encoding="utf-8"
    ) as dpo_f:
        for i in range(repeats_per_scenario):
            for scenario in SCENARIOS:
                try:
                    data = generate_batch(scenario)
                    sft_row = {
                        "messages": [
                            {"role": "system", "content": MENTOR_SYSTEM_PROMPT},
                            {"role": "user", "content": scenario},
                            {"role": "assistant", "content": data["sft_response"]},
                        ]
                    }
                    dpo_row = {
                        "prompt": scenario,
                        "chosen": data["dpo_chosen"],
                        "rejected": data["dpo_rejected"],
                    }
                    sft_f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
                    dpo_f.write(json.dumps(dpo_row, ensure_ascii=False) + "\n")
                    sft_count += 1
                    dpo_count += 1
                    print(f"[Pass {i+1}] Saved example #{sft_count}")
                except Exception as e:
                    print(f"Skipped scenario due to error: {e}")

    print(f"\nDone! Generated {sft_count} SFT rows and {dpo_count} DPO pairs.")


if __name__ == "__main__":
    # Start with repeats_per_scenario=1 (10 examples) to test cost/quality, then scale to 30+ (300+ rows)
    build_datasets(repeats_per_scenario=8)