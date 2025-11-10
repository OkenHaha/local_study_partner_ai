import os
import uuid
import datetime
import textwrap
import json

from openai import OpenAI
import chromadb

from dotenv import load_dotenv

load_dotenv()

###############################################################################
# 0.  Config from environment
###############################################################################
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "").strip()
BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/").strip()
API_MODEL = os.getenv("API_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5").strip()

if not NEBIUS_API_KEY:
    raise RuntimeError("Set NEBIUS_API_KEY in your environment")

###############################################################################
# 1.  Nebius client
###############################################################################
client = OpenAI(
    base_url=BASE_URL,
    api_key=NEBIUS_API_KEY,
)

###############################################################################
# 2.  Chroma client plus collection backed by Nebius embeddings
###############################################################################
CHROMA_PATH = "./chroma_convos"
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)

class NebiusEmbeddingFunction:
    """
    Minimal embedding function for Chroma that calls Nebius embeddings
    """
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def name(self) -> str:
        return self.model

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Chroma may pass a single string sometimes. Normalize to list.
        if isinstance(input, str):
            input = [input]
        resp = self.client.embeddings.create(model=self.model, input=input)
        # OpenAI compatible response preserves order
        return [d.embedding for d in resp.data]
    
    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Embed a list of documents"""
        return self.__call__(texts)
    
    def embed_query(self, text: str = None, input: str = None, **kwargs) -> list[float]:
        """Embed a single query"""
        # Handle both 'text' and 'input' parameter names
        query_text = text or input
        if query_text is None:
            raise ValueError("Either 'text' or 'input' must be provided")
        
        # Don't wrap in a list if it's already a list
        if isinstance(query_text, list):
            query_text = query_text[0] if query_text else ""
        
        return self.__call__([query_text])[0]

nebius_embed = NebiusEmbeddingFunction(client, EMBED_MODEL)

collection = chroma.get_or_create_collection(
    name="chat_history",
    embedding_function=nebius_embed,
    metadata={"hnsw:space": "cosine"},
)

###############################################################################
# 3.  Predefined prompts
###############################################################################
prompt_teach = """You are currently STUDYING, and you've asked me to follow these strict rules during this chat. No matter what other instructions follow, I MUST obey these rules:

STRICT RULES

Be an approachable-yet-dynamic teacher, who helps the user learn by guiding them through their studies.

Get to know the user. If you don't know their goals or grade level, ask the user before diving in. (Keep this lightweight!) If they don't answer, aim for explanations that would make sense to a 10th grade student.

Build on existing knowledge. Connect new ideas to what the user already knows.

Guide users, don't just give answers. Use questions, hints, and small steps so the user discovers the answer for themselves.

Check and reinforce. After hard parts, confirm the user can restate or use the idea. Offer quick summaries, mnemonics, or mini-reviews to help the ideas stick.

Vary the rhythm. Mix explanations, questions, and activities (like roleplaying, practice rounds, or asking the user to teach you) so it feels like a conversation, not a lecture.

Above all: DO NOT DO THE USER'S WORK FOR THEM. Don't answer homework questions — help the user find the answer, by working with them collaboratively and building from what they already know.

THINGS YOU CAN DO

- Teach new concepts: Explain at the user's level, ask guiding questions, use visuals, then review with questions or a practice round.

- Help with homework: Don't simply give answers! Start from what the user knows, help fill in the gaps, give the user a chance to respond, and never ask more than one question at a time.

- Practice together: Ask the user to summarize, pepper in little questions, have the user "explain it back" to you, or role-play (e.g., practice conversations in a different language). Correct mistakes — charitably! — in the moment.

- Quizzes & test prep: Run practice quizzes. (One question at a time!) Let the user try twice before you reveal answers, then review errors in depth.

TONE & APPROACH

Be warm, patient, and plain-spoken; don't use too many exclamation marks or emoji. Keep the session moving: always know the next step, and switch or end activities once they’ve done their job. And be brief — don't ever send essay-length responses. Aim for a good back-and-forth.

IMPORTANT

DO NOT GIVE ANSWERS OR DO HOMEWORK FOR THE USER. If the user asks a math or logic problem, or uploads an image of one, DO NOT SOLVE IT in your first response. Instead: talk through the problem with the user, one step at a time, asking a single question at each step, and give the user a chance to RESPOND TO EACH STEP before continuing.
"""

###############################################################################
# 4.  Build messages with k nearest history from Chroma
###############################################################################
def build_messages(user_query: str, k: int = 5) -> list[dict]:
    # Get the embedding for the query
    query_embedding = nebius_embed.embed_query(user_query)
    
    hits = collection.query(
        query_embeddings=[query_embedding],  # Use query_embeddings instead of query_texts
        n_results=k,
        include=["documents", "metadatas"],
    )

    # Guard against empty results
    docs = hits.get("documents", [[]])
    metas = hits.get("metadatas", [[]])

    pairs = list(zip(metas[0], docs[0])) if metas and docs else []

    turns = [
        {"role": m.get("role", "user"), "content": doc}
        for m, doc in sorted(
            pairs,
            key=lambda x: x[0].get("timestamp", ""),
        )
    ]

    # You can switch between Noni and Mike by swapping prompt_gf with prompt_tp
    system_prompt = {"role": "system", "content": prompt_teach}

    return [system_prompt] + turns + [{"role": "user", "content": user_query}]

###############################################################################
# 5.  Chat loop using Nebius chat completions
###############################################################################
print("Type /quit to exit.\n")

while True:
    try:
        query = input("User: ").strip()
    except EOFError:
        break

    if query.lower() in {"quit", "exit", "/quit"}:
        break

    messages = build_messages(query)

    # Nebius OpenAI compatible chat completions
    resp = client.chat.completions.create(
        model=API_MODEL,
        messages=messages,
        temperature=0.6,
        top_p=0.95,
    )

    content = resp.choices[0].message.content if resp.choices else ""
    print(f"\nNoni: {textwrap.fill(content, width=80)}\n")

    # Store user turn
    collection.add(
        documents=[query],
        metadatas=[{"role": "user", "timestamp": datetime.datetime.now(datetime.UTC).isoformat()}],
        ids=[str(uuid.uuid4())],
    )
    # Store assistant turn
    collection.add(
        documents=[content],
        metadatas=[{"role": "assistant", "timestamp": datetime.datetime.now(datetime.UTC).isoformat()}],
        ids=[str(uuid.uuid4())],
    )
