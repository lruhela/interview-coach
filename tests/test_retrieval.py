import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever import answer_question

# Test with a real interview question
question = "Tell me about a time you led a large cross-functional initiative"
result = answer_question(question)

print(f"\n🎯 Question: {result['question']}")
print(f"\n📚 Retrieved Stories:")
for s in result['retrieved_stories']:
    print(f"  - {s['id']}: {s['metadata']['title']} (relevance: {1 - s['distance']:.2f})")

print(f"\n🤖 Recommendation:\n{result['recommendation']}")