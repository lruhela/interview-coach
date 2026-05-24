import os
import chromadb
import voyageai
from groq import Groq
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(__file__))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

vc            = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
chroma_client = chromadb.PersistentClient(path=os.path.join(project_root, ".chroma"))
collection    = chroma_client.get_collection(name="star_stories")
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))

_LLM = "llama-3.3-70b-versatile"


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
    """Expand query for richer semantic match, then search ChromaDB."""
    expanded = _expand_query(question)

    query_embedding = vc.embed(
        [expanded],
        model="voyage-3",       # upgraded from voyage-3-lite for better accuracy
        input_type="query",
    ).embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    stories = []
    for i in range(len(results["ids"][0])):
        stories.append({
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return stories


def generate_recommendation(question: str, stories: list[dict]) -> str:
    """Coaching analysis: which story to use, why, gaps, and what to emphasize."""
    context = ""
    for i, s in enumerate(stories, 1):
        context += f"\nStory {i} [{s['id']}]: {s['document']}\nMatch score: {1 - s['distance']:.2f}\n---"

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
        max_tokens=500,
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
