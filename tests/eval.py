"""
Retrieval evaluation — tests whether the right stories surface for known questions.

💡 CONCEPT — Why evaluate RAG?
In a production system you'd have hundreds of test cases and track
retrieval precision/recall over time. We're doing a small version of
the same thing. The principle is identical.

Precision = of the stories retrieved, how many were actually relevant?
Recall = of the relevant stories, how many did we retrieve?
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retriever import retrieve_stories

# Ground truth: for each question, which story IDs SHOULD surface?
# You define this based on your own knowledge of your stories.
EVAL_SET = [
    {
        "question": "Tell me about a time you led a large cross-functional initiative",
        "expected_top": "S1",
        "acceptable": ["S1", "S6", "S7"]
    },
    {
        "question": "Tell me about a failure or something you would do differently",
        "expected_top": "S4",
        "acceptable": ["S4", "S12"]
    },
    {
        "question": "How do you handle conflict between two senior engineers?",
        "expected_top": "S11",
        "acceptable": ["S11", "S9"]
    },
    {
        "question": "Tell me about a hard data-driven decision you made",
        "expected_top": "S5",
        "acceptable": ["S5", "S8"]
    },
    {
        "question": "How do you grow and develop engineers on your team?",
        "expected_top": "S10",
        "acceptable": ["S10"]
    },
    {
        "question": "Tell me about a platform or infrastructure system you built",
        "expected_top": "S3",
        "acceptable": ["S3", "S6", "S2"]
    },
    {
        "question": "Tell me about a time you had to influence without authority",
        "expected_top": "S8",
        "acceptable": ["S8", "S12"]
    },
    {
        "question": "Describe a time you led through resistance or a difficult change",
        "expected_top": "S7",
        "acceptable": ["S7", "S4"]
    }
]

def run_eval():
    print("🧪 Running retrieval evaluation...\n")
    
    passed = 0
    top_hit = 0
    total = len(EVAL_SET)

    for test in EVAL_SET:
        question = test["question"]
        expected_top = test["expected_top"]
        acceptable = test["acceptable"]

        results = retrieve_stories(question, n_results=3)
        retrieved_ids = [r["id"] for r in results]
        top_result = retrieved_ids[0] if retrieved_ids else None

        top_correct = top_result == expected_top
        any_correct = any(r in acceptable for r in retrieved_ids)

        if top_correct:
            top_hit += 1
        if any_correct:
            passed += 1

        status = "✅" if top_correct else ("🟡" if any_correct else "❌")
        print(f"{status} Q: {question[:60]}...")
        print(f"   Expected top: {expected_top} | Got: {retrieved_ids}")
        print()

    print("=" * 60)
    print(f"Top-1 accuracy:  {top_hit}/{total} ({int(top_hit/total*100)}%)")
    print(f"Any-hit accuracy: {passed}/{total} ({int(passed/total*100)}%)")
    print()
    
    if top_hit / total < 0.7:
        print("⚠️  Top-1 below 70% — check trigger_questions in stories.json")
    else:
        print("🎉 Retrieval quality is solid")

if __name__ == "__main__":
    run_eval()