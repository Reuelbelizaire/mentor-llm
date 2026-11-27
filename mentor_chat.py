from openai import OpenAI
from memory_db import init_db, add_note, get_memory_context, save_turn

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  
)

def ask_mentor(user_message: str) -> str:
    notes_block, recent_history = get_memory_context(limit_messages=4)

    system_context = (
        "You are Reuel's Senior Data Scientist mentor inside The Sanctuary. "
        "Be concise, direct, slightly witty, and evaluate ideas using Simplicity, Reliability, and Impact.\n\n"
        f"Saved Memory & Active Projects:\n{notes_block}"
    )

    messages = [{"role": "system", "content": system_context}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="senior-ds-mentor",
        messages=messages,
        temperature=0.3,
    )

    reply = response.choices[0].message.content.strip()
    save_turn("user", user_message)
    save_turn("assistant", reply)
    return reply

if __name__ == "__main__":
    init_db()
    print("The Sanctuary CLI is live! Type '/note <text>' to save memory, or 'exit' to quit.\n")
    while True:
        user_in = input("You: ").strip()
        if not user_in or user_in.lower() == "exit":
            break
        if user_in.startswith("/note "):
            add_note("Manual Note", user_in[6:].strip())
            print("Mentor: Saved to memory.\n")
            continue
        print(f"\nMentor: {ask_mentor(user_in)}\n")