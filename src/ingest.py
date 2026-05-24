import json
import os
import chromadb
import voyageai
from dotenv import load_dotenv

# Always resolve paths relative to this file, not the working directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

vc = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Absolute path so this works regardless of which directory you run from
chroma_client = chromadb.PersistentClient(path=os.path.join(project_root, ".chroma"))


def load_stories(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def build_document(story: dict) -> str:
    """
    Combines all story fields into one text block for embedding.

    💡 CONCEPT — Chunking strategy:
    This is one of the most important RAG design decisions.
    We're embedding the ENTIRE story as one chunk because:
    - Each story is self-contained and needs full context to be retrieved correctly
    - Splitting mid-story would lose meaning (a Result without its Action is useless)
    - Our stories are short enough that one chunk per story works well

    In larger RAG systems (e.g. ingesting 100-page documents), you'd
    split into smaller overlapping chunks. The tradeoff is always:
    smaller chunks = more precise retrieval, but less context per chunk.

    💡 DESIGN NOTE — trigger_questions lead the document:
    We put trigger_questions first because they are the strongest retrieval
    signal. Embedding models weight earlier tokens more heavily, so leading
    with the curated interview question patterns improves recall for those
    specific phrasings.
    """
    triggers = ', '.join(story['trigger_questions'])
    themes   = ', '.join(story['themes'])
    return (
        f"Best for questions about: {triggers}\n"
        f"Key themes: {themes}\n\n"
        f"Title: {story['title']}\n"
        f"Situation: {story['situation']}\n"
        f"Task: {story['task']}\n"
        f"Action: {story['action']}\n"
        f"Result: {story['result']}"
    )


def ingest_stories(path: str | None = None):
    if path is None:
        path = os.path.join(project_root, "data", "stories.json")

    stories = load_stories(path)

    # ── Reset collections so distance metric changes take effect ─────────────
    # We delete-and-recreate because get_or_create_collection will NOT update
    # the distance metric on an existing collection. Ingest is always a full
    # refresh, so losing the old data is fine.
    for name in ("star_stories", "star_trigger_questions"):
        try:
            chroma_client.delete_collection(name)
        except Exception:
            pass  # didn't exist yet — that's fine

    # 💡 CONCEPT — cosine vs L2 distance:
    # ChromaDB defaults to L2 (Euclidean) distance. For normalized embeddings
    # (which Voyage AI produces), cosine distance is more intuitive:
    #   - distance = 0   → identical meaning
    #   - distance = 1   → orthogonal (unrelated)
    #   - distance = 2   → opposite
    # This makes the "match %" display in the UI straightforward:
    #   pct = (1 - distance) * 100  gives values in roughly [50%, 100%] for
    #   semantically relevant results.
    story_collection = chroma_client.create_collection(
        name="star_stories",
        metadata={"hnsw:space": "cosine"},
    )
    trigger_collection = chroma_client.create_collection(
        name="star_trigger_questions",
        metadata={"hnsw:space": "cosine"},
    )

    # ── Full-story embeddings ─────────────────────────────────────────────────
    documents = []
    metadatas = []
    ids = []

    for story in stories:
        doc = build_document(story)
        documents.append(doc)
        metadatas.append({
            "id": story["id"],
            "title": story["title"],
            "themes": ", ".join(story["themes"]),
            "best_for_roles": ", ".join(story["best_for_roles"]),
        })
        ids.append(story["id"])

    # Batch embed all documents in one API call — more efficient than one by one
    # voyage-3 provides better semantic accuracy than voyage-3-lite
    result = vc.embed(documents, model="voyage-3", input_type="document")

    story_collection.upsert(
        documents=documents,
        embeddings=result.embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    # ── Trigger-question embeddings ───────────────────────────────────────────
    # 💡 KEY DESIGN DECISION — two-collection retrieval:
    #
    # Problem: a trigger question like "tell me about a failure" is a dozen
    # words embedded inside a 300-word story document. The long narrative
    # dominates the vector, so the trigger signal gets diluted.
    #
    # Fix: index every trigger question as its own document in a second
    # collection. At retrieval time we search BOTH collections. If a user's
    # question is close to any trigger question, that story gets boosted even
    # when the full story text is less obviously relevant.
    #
    # The metadata.story_id link lets us merge the two result sets.
    tq_documents = []
    tq_metadatas = []
    tq_ids = []

    for story in stories:
        for idx, tq in enumerate(story["trigger_questions"]):
            tq_documents.append(tq)
            tq_metadatas.append({
                "story_id": story["id"],
                "title": story["title"],
            })
            tq_ids.append(f"{story['id']}_tq_{idx}")

    # Use the same voyage-3 model for consistency — all embeddings must share
    # the same model so their vectors live in the same space.
    tq_result = vc.embed(tq_documents, model="voyage-3", input_type="document")

    trigger_collection.upsert(
        documents=tq_documents,
        embeddings=tq_result.embeddings,
        metadatas=tq_metadatas,
        ids=tq_ids,
    )

    print(f"✅ Ingested {len(stories)} stories into ChromaDB")
    print(f"   Embedding dimensions : {len(result.embeddings[0])}")
    print(f"   Stories stored       : {[s['id'] for s in stories]}")
    print(f"   Trigger questions    : {len(tq_documents)} total across all stories")


if __name__ == "__main__":
    ingest_stories()
