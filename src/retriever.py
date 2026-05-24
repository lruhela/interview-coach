import os
import chromadb
import voyageai
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the project root
project_root = os.path.dirname(os.path.dirname(__file__))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

vc = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path=os.path.join(project_root, ".chroma"))

# Load collections lazily — get_collection() raises if ingest hasn't run,
# and we don't want to crash the whole app at import time.
def _get_collection(name: str):
    try:
        return chroma_client.get_collection(name=name)
    except Exception:
        return None

_story_collection = None
_trigger_collection = None

def _collections():
    """Return (story_collection, trigger_collection), loading once on first call."""
    global _story_collection, _trigger_collection
    if _story_collection is None:
        _story_collection = _get_collection("star_stories")
    if _trigger_collection is None:
        _trigger_collection = _get_collection("star_trigger_questions")
    return _story_collection, _trigger_collection


def retrieve_stories(question: str, n_results: int = 3) -> list[dict]:
    """
    Dual-collection semantic search.

    💡 CONCEPT — Why two collections?

    Traditional single-collection RAG embeds the entire story as one document.
    The problem: a 300-word story narrative dominates its embedding vector,
    drowning out the 10-12 short trigger question phrases at the end.
    Result: "Tell me about a failure" might not surface S4 even though S4's
    trigger_questions explicitly contains "tell me about a failure".

    Fix — run two parallel searches:
    1. star_stories: full story text → captures broad semantic meaning
    2. star_trigger_questions: one embedding per trigger question → captures
       exact question-pattern matches

    Then merge by story_id, giving trigger matches a score boost. Stories that
    closely match a curated trigger question surface even when their narrative
    text is less obviously relevant to the question phrasing.

    💡 CONCEPT — Semantic search vs keyword search:
    Traditional search matches exact words. If you search "scaling infrastructure"
    it won't find a story that says "improved platform reliability" even though
    they mean similar things.

    Semantic search converts both your question AND your stories into vectors,
    then finds stories whose vectors are mathematically closest to your question.
    "Scaling infrastructure" and "improved platform reliability" end up near
    each other in vector space because they share meaning.
    """
    story_col, trigger_col = _collections()

    if story_col is None:
        raise RuntimeError(
            "Story collection not found. Run `python src/ingest.py` first to load your stories."
        )

    query_embedding = vc.embed(
        [question],
        model="voyage-3-lite",
        input_type="query",  # "query" vs "document" optimizes the embedding differently
    ).embeddings[0]

    # ── 1. Full-story semantic search (fetch ALL stories so we rank ourselves) ──
    total_stories = story_col.count()
    story_results = story_col.query(
        query_embeddings=[query_embedding],
        n_results=total_stories,
        include=["documents", "metadatas", "distances"],
    )

    # Build a map: story_id → {doc, metadata, story_distance, trigger_distance}
    # story_distance  = cosine distance of full story text
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

    # ── 2. Trigger-question search (if that collection exists) ────────────────
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

    # ── 3. Combine scores ─────────────────────────────────────────────────────
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

    # ── 4. Sort and return top n_results ──────────────────────────────────────
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
    """
    Takes retrieved stories and asks the LLM to recommend which to use and why.

    💡 CONCEPT — This is the 'Generation' in Retrieval Augmented Generation.
    We're not asking the LLM to answer from its training data.
    We're giving it YOUR stories as context and asking it to reason over them.
    This is what prevents hallucination — the LLM can only work with what we provide.
    """

    # Build context block from retrieved stories
    # Relevance: cosine_similarity = 1 - distance, range [−1, 1].
    # For good matches distance is well below 1, so similarity is positive and intuitive.
    context = ""
    for i, story in enumerate(stories, 1):
        similarity = max(0.0, 1.0 - story["distance"])
        context += f"\nStory {i} [{story['id']}]: {story['document']}\n"
        context += f"Relevance: {similarity:.2f}\n"
        context += "---"

    prompt = f"""You are an expert interview coach helping an experienced engineering leader prepare for interviews.

The candidate has been asked this interview question:
"{question}"

Here are their most relevant STAR stories based on semantic search:
{context}

Your job:
1. Recommend which story (or stories) to lead with and why
2. Identify what specific details from the story directly answer the question
3. Flag any gaps — parts of the question the story doesn't cover well
4. Suggest one concrete thing they should emphasize or add when telling this story

Be direct and specific. This person is a Director-level candidate — treat them as one."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def answer_question(question: str, n_results: int = 3) -> dict:
    """Main entry point — retrieve then generate."""
    stories = retrieve_stories(question, n_results=n_results)
    recommendation = generate_recommendation(question, stories)

    return {
        "question": question,
        "retrieved_stories": stories,
        "recommendation": recommendation,
    }
