import gradio as gr
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retriever import answer_question

def coach(question, n_results):
    if not question.strip():
        return "Please enter an interview question.", ""

    result = answer_question(question, n_results=int(n_results))

    # Format retrieved stories
    stories_output = ""
    for i, story in enumerate(result["retrieved_stories"], 1):
        relevance = int((1 - story["distance"]) * 100)
        stories_output += f"### #{i} — {story['metadata']['title']} ({relevance}% match)\n"
        stories_output += f"**Themes:** {story['metadata']['themes']}\n\n"
        stories_output += f"{story['document']}\n\n---\n\n"

    return stories_output, result["recommendation"]

with gr.Blocks(title="Interview Coach") as app:
    gr.Markdown("# 🎯 Interview Coach")
    gr.Markdown("Paste an interview question. Get your best STAR stories and coaching.")

    with gr.Row():
        question = gr.Textbox(
            label="Interview Question",
            placeholder="e.g. Tell me about a time you led a cross-functional initiative...",
            lines=3
        )

    n_results = gr.Slider(
        minimum=1, maximum=5, value=3, step=1,
        label="Number of stories to retrieve"
    )

    submit = gr.Button("Find My Best Stories", variant="primary")

    with gr.Row():
        with gr.Column():
            stories_out = gr.Markdown(label="📚 Retrieved Stories")
        with gr.Column():
            coaching_out = gr.Markdown(label="🤖 Coaching Recommendation")

    submit.click(
        fn=coach,
        inputs=[question, n_results],
        outputs=[stories_out, coaching_out]
    )

app.launch()