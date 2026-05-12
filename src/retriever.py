import os
import chromadb
import voyageai
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the project root
project_root = os.path.dirname(os.path.dirname(__file__))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

vc = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
chroma_client = chromadb.PersistentClient(path=os.path.join(project_root, ".chroma"))
collection = chroma_client.get_collection(name="star_stories")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve_stories(question: str, n_results: int = 3) -> list[dict]:
    """
    💡 CONCEPT — Semantic search vs keyword search:
    Traditional search matches exact words. If you search "scaling infrastructure"
    it won't find a story that says "improved platform reliability" even though
    they mean similar things.

    Semantic search converts both your question AND your stories into vectors,
    then finds stories whose vectors are mathematically closest to your question.
    "Scaling infrastructure" and "improved platform reliability" end up near
    each other in vector space because they share meaning.

    This is why RAG is powerful for unstructured content like STAR stories.
    """
    query_embedding = vc.embed(
        [question],
        model="voyage-3-lite",
        input_type="query"  # important: "query" vs "document" optimizes the embedding differently
    ).embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    stories = []
    for i in range(len(results['ids'][0])):
        stories.append({
            "id": results['ids'][0][i],
            "document": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
            # 💡 distance = how far apart the vectors are
            # Lower distance = more semantically similar = better match
        })

    return stories

def generate_recommendation(question: str, stories: list[dict]) -> str:
    """
    Takes retrieved stories and asks the LLM to recommend which to use and why.

    💡 CONCEPT — This is the 'Generation' in Retrieval Augmented Generation.
    We're not asking the LLM to answer from its training data.
    We're giving it YOUR stories as context and asking it to reason over them.
    This is what prevents hallucination — the LLM can only work with what we provide.
    """

    # Build context block from retrieved stories
    context = ""
    for i, story in enumerate(stories, 1):
        context += f"\nStory {i} [{story['id']}]: {story['document']}\n"
        context += f"Relevance score: {1 - story['distance']:.2f}\n"
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
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def answer_question(question: str, n_results: int = 3) -> dict:
    """Main entry point — retrieve then generate."""
    stories = retrieve_stories(question, n_results=n_results)
    recommendation = generate_recommendation(question, stories)

    return {
        "question": question,
        "retrieved_stories": stories,
        "recommendation": recommendation
    }