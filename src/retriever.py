# -*- coding: utf-8 -*-
import os
import chromadb
import voyageai
from groq import Groq
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(__file__))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

vc          = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_LLM = "llama-3.3-70b-versatile"

chroma_client = chromadb.PersistentClient(path=os.path.join(project_root, ".chroma"))

# Load collections lazily — get_collection() raises if ingest hasn't run,
# and we don't want to crash the whole app at import time.
def _get_collection(name: str):
    try:
        return chroma_client.get_collection(name=name)
    except Exception:
        return None

_story_collection   = None
_trigger_collection = None

def _collections():
    """Return (story_collection, trigger_collection), loading once on first call."""
    global _story_collection, _trigger_collection
    if _story_collection is None:
        _story_collection = _get_collection("star_stories")
    if _trigger_collection is None:
        _trigger_collection = _get_collection("star_trigger_questions")
    return _story_collection, _trigger_collection


def _expand_query(question: str) -> str:
    """
    Expand the raw interview question with related themes and synonyms before
    embedding, so the query vector covers more semantic surface area.

    💡 CONCEPT — Query expansion:
    "Tell me about promoting someone" and "talent development / sponsorship /
    career growth" mean the same thing but live in different parts of the vector
    space. Expanding the query bridges that gap without changing the stored docs.
    """
    resp = groq_client.chat.completions.create(
        model=_LLM,
        max_tokens=80,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": (
                f'Interview question: "{question}"\n\n'
                "List 6-8 key leadership themes, skills, and synonyms this question "
                "is testing. Output only a comma-separated list — no explanation."
            ),
        }],
    )
    expansion = resp.choices[0].message.content.strip()
    return f"{question}\n{expansion}"


def retrieve_stories(question: str, n_results: int = 3) -> list[dict]:
    """
    Three-layer retrieval: query expansion → dual-collection search → score fusion.

    💡 CONCEPT — Why three layers?

    Layer 1 — Query expansion:
    "Tell me about a hard decision" and "data-driven decision making, risk
    management, tradeoffs" share meaning but live far apart in vector space.
    An LLM expands the bare question with related themes so the query vector
    covers more semantic ground before we even hit the database.

    Layer 2 — Dual-collection search:
    Traditional single-collection RAG embeds the full story as one document.
    The problem: a 300-word narrative dominates its embedding vector, drowning
    out the 10-12 short trigger question phrases. "Tell me about a failure"
    might not surface S4 even though S4's trigger_questions explicitly contains
    "tell me about a failure".

    Fix — search two collections in parallel:
    - star_stories: full story text → broad semantic meaning
    - star_trigger_questions: one embedding per trigger phrase → exact pattern match

    Layer 3 — Score fusion:
    combined = min(trigger_dist × 0.65, story_dist)
    Trigger matches win when strong; story distance wins otherwise.
    """
    story_col, trigger_col = _collections()

    if story_col is None:
        raise RuntimeError(
            "Story collection not found. Run `python src/ingest.py` first to load your stories."
        )

    # ── 1. Query expansion ────────────────────────────────────────────────────
    expanded = _expand_query(question)

    # ── 2. Embed the expanded query ───────────────────────────────────────────
    query_embedding = vc.embed(
        [expanded],
        model="voyage-3",       # upgraded from voyage-3-lite for better accuracy
        input_type="query",     # "query" vs "document" optimizes the embedding differently
    ).embeddings[0]

    # ── 3. Full-story semantic search (fetch ALL stories so we rank ourselves) ──
    total_stories = story_col.count()
    story_results = story_col.query(
        query_embeddings=[query_embedding],
        n_results=total_stories,
        include=["documents", "metadatas", "distances"],
    )

    # Build a map: story_id → {doc, metadata, story_distance, trigger_distance}
    # story_distance   = cosine distance of full story text
    # trigger_distance = cosine distance of the best-matching trigger question
    story_map: dict[str, dict] = {}
    for i in range(len(story_results["ids"][0])):
        sid = story_results["ids"][0][i]
        story_map[sid] = {
            "id": sid,
            "document": story_results["documents"][0][i],
            "metadata": story_results["metadatas"][0][i],
            "story_distance": story_results["distances"][0][i],
            "trigger_distance": float("inf"),  # will be updated below
        }

    # ── 4. Trigger-question search (if that collection exists) ────────────────
    if trigger_col is not None:
        total_triggers = trigger_col.count()
        # Fetch enough to cover all stories (each story has ~10 trigger questions)
        tq_results = trigger_col.query(
            query_embeddings=[query_embedding],
            n_results=min(total_triggers, 50),
            include=["metadatas", "distances"],
        )

        # For each trigger-question hit, record the *best* (lowest) distance
        # for that story so we know how close the question got to any trigger.
        for i in range(len(tq_results["ids"][0])):
            sid = tq_results["metadatas"][0][i]["story_id"]
            tdist = tq_results["distances"][0][i]
            if sid in story_map and tdist < story_map[sid]["trigger_distance"]:
                story_map[sid]["trigger_distance"] = tdist

    # ── 5. Score fusion ───────────────────────────────────────────────────────
    # 💡 CONCEPT — Score fusion:
    # We have two distance signals per story. We want:
    #   - Trigger match to win when it's strong (user asked a question very
    #     close to a curated trigger pattern)
    #   - Story match to win when there's no trigger match (broader paraphrase)
    #
    # Formula:  combined = min(trigger_dist * TRIGGER_WEIGHT, story_dist)
    #
    # TRIGGER_WEIGHT = 0.65 means a trigger distance of 0.15 becomes 0.098,
    # which beats a story-only distance of 0.12. The trigger signal wins.
    # If there's no trigger match (trigger_dist = inf), combined = story_dist.
    TRIGGER_WEIGHT = 0.65

    for s in story_map.values():
        sd = s["story_distance"]
        td = s["trigger_distance"]
        s["distance"] = min(td * TRIGGER_WEIGHT, sd) if td < float("inf") else sd

    # ── 6. Sort and return top n_results ──────────────────────────────────────
    # Lower combined distance = better match
    ranked = sorted(story_map.values(), key=lambda x: x["distance"])[:n_results]

    return [
        {
            "id": s["id"],
            "document": s["document"],
            "metadata": s["metadata"],
            "distance": s["distance"],
            # 💡 distance = cosine distance (0 = identical, 2 = opposite)
            # Lower is better. The UI converts this to a match percentage.
        }
        for s in ranked
    ]


def generate_recommendation(question: str, stories: list[dict]) -> str:
    """Coaching analysis: which story to use, why, gaps, and what to emphasize."""
    context = ""
    for i, story in enumerate(stories, 1):
        # Clamp to [0, 1] — cosine_similarity = 1 - distance, but can be slightly
        # negative for very dissimilar vectors; floor at 0 for display clarity.
        similarity = max(0.0, 1.0 - story["distance"])
        context += f"\nStory {i} [{story['id']}]: {story['document']}\nMatch score: {similarity:.2f}\n---"

    prompt = f"""You are an expert interview coach helping a Director-level engineering leader prepare.

Interview question: "{question}"

Retrieved stories:
{context}

Provide a direct, specific coaching analysis:
1. Which story to lead with and exactly why it fits this question
2. Specific details from the story that directly answer what's being asked
3. Any gaps — aspects of the question the story doesn't cover well
4. One concrete thing to emphasize or add when telling this story

Be direct. Treat them as the Director-level candidate they are."""

    resp = groq_client.chat.completions.create(
        model=_LLM,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def generate_polished_story(question: str, story: dict) -> str:
    """
    Generate a polished, practice-ready first-person STAR narrative
    tailored to the specific interview question.
    """
    prompt = f"""You are an expert interview coach preparing a Director-level engineering leader for an interview.

Interview question: "{question}"

Their STAR story to use:
{story['document']}

Write a polished, first-person spoken version of this story that:
- Opens with a strong, direct hook addressing the question (no "Sure, let me tell you about...")
- Follows STAR structure naturally, without labeling the sections
- Uses confident, precise language fitting a Director-level leader
- Runs approximately 2 minutes when spoken aloud (250–320 words)
- Closes with measurable impact and a brief forward-looking insight

Format as flowing paragraphs ready to read aloud and practice. No headers, no bullet points, no markdown."""

    resp = groq_client.chat.completions.create(
        model=_LLM,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def answer_question(question: str, n_results: int = 3) -> dict:
    """Main entry point: retrieve → recommend → polish."""
    stories        = retrieve_stories(question, n_results=n_results)
    recommendation = generate_recommendation(question, stories)

    # Polished story uses the top semantic match
    top_story = sorted(stories, key=lambda s: s["distance"])[0]
    polished  = generate_polished_story(question, top_story)

    return {
        "question":          question,
        "retrieved_stories": stories,
        "recommendation":    recommendation,
        "polished_story":    polished,
    }
