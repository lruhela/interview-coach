import json
import os
import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

# Initialize Voyage for embeddings
vc = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Initialize ChromaDB — persists to disk so you don't re-embed every run
chroma_client = chromadb.PersistentClient(path=".chroma")
collection = chroma_client.get_or_create_collection(name="star_stories")

def load_stories(path: str) -> list[dict]:
    with open(path, 'r') as f:
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

def ingest_stories(path: str = "data/stories.json"):
    stories = load_stories(path)

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
            "best_for_roles": ", ".join(story["best_for_roles"])
        })
        ids.append(story["id"])

    # Batch embed all documents in one API call — more efficient than one by one
    # voyage-3 provides better semantic accuracy than voyage-3-lite
    result = vc.embed(documents, model="voyage-3", input_type="document")

    # Upsert = insert if new, update if ID already exists
    # This means you can re-run ingest safely after editing stories
    collection.upsert(
        documents=documents,
        embeddings=result.embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Ingested {len(stories)} stories into ChromaDB")
    print(f"   Embedding dimensions: {len(result.embeddings[0])}")
    print(f"   Stories stored: {[s['id'] for s in stories]}")

if __name__ == "__main__":
    ingest_stories()