import openai
from groq import Groq
import voyageai
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

# Test Voyage embeddings
vc = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
result = vc.embed(["test story to prepare for the interviews"], model="voyage-3-lite")
print(f"✅ Voyage embedding dims: {len(result.embeddings[0])}")

# Test ChromaDB
client = chromadb.PersistentClient(path=".chroma")
col = client.get_or_create_collection("test")
print(f"✅ ChromaDB working")

# Test OpenAI
# oc = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# msg = oc.chat.completions.create(
#    model="gpt-4o-mini",  # cheapest model, good for dev
#    max_tokens=50,
#    messages=[{"role": "user", "content": "Say 'setup complete' and nothing else"}]
#)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # cheapest model, good for dev
    max_tokens=50,
    messages=[{"role": "user", "content": "Say 'setup complete' and nothing else"}]
)
print(f"✅ GROQ: {response.choices[0].message.content}")