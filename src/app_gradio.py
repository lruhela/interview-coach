import gradio as gr
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retriever import answer_question

# ── Design tokens ─────────────────────────────────────────────────────────────
FONT  = ("-apple-system,BlinkMacSystemFont,'SF Pro Display','SF Pro Text',"
         "'Helvetica Neue','Segoe UI',Arial,sans-serif")
BLUE  = "#0071e3"
BG    = "#f5f5f7"
DARK  = "#1d1d1f"
MID   = "#6e6e73"
LIGHT = "#86868b"
WHITE = "white"

# ── Gradio theme: force light mode for both light AND dark system preferences ──
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
).set(
    # Light mode
    body_background_fill=BG,
    background_fill_primary=WHITE,
    background_fill_secondary=BG,
    block_background_fill=WHITE,
    input_background_fill=WHITE,
    border_color_primary="#d2d2d7",
    block_border_width="0px",
    block_shadow="none",
    # Dark mode — intentionally identical to force light appearance
    body_background_fill_dark=BG,
    background_fill_primary_dark=WHITE,
    background_fill_secondary_dark=BG,
    block_background_fill_dark=WHITE,
    input_background_fill_dark=WHITE,
    border_color_primary_dark="#d2d2d7",
)

# ── CSS: fine-tuning on top of theme ──────────────────────────────────────────
APPLE_CSS = f"""
/* 1 ─ Font stack: SF Pro → Helvetica Neue → system fallback */
html, body, *, *::before, *::after {{
    font-family: {FONT} !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    box-sizing: border-box;
}}

/* 2 ─ Force light background even when OS is in dark mode */
html, body, .dark, html.dark, html.dark body {{
    background: {BG} !important;
    color: {DARK} !important;
}}
.dark .gradio-container, .gradio-container {{
    background: {BG} !important;
}}

/* 3 ─ Page layout */
.gradio-container {{
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 48px 24px 80px !important;
}}

/* 4 ─ Hide Gradio footer */
footer, .footer {{ display: none !important; }}

/* 5 ─ Strip noisy block chrome */
.block, .form {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin-bottom: 16px !important;
}}

/* 6 ─ Text inputs: light background, blue focus ring */
textarea, input[type="text"], input[type="search"],
.dark textarea, .dark input[type="text"] {{
    font-size: 15px !important;
    font-weight: 400 !important;
    color: {DARK} !important;
    background: #fafafa !important;
    border: 1.5px solid #d2d2d7 !important;
    border-radius: 12px !important;
    padding: 13px 16px !important;
    line-height: 1.55 !important;
    box-shadow: none !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    resize: none !important;
}}
textarea:focus, input[type="text"]:focus,
.dark textarea:focus, .dark input:focus {{
    border-color: {BLUE} !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.14) !important;
    outline: none !important;
    background: {WHITE} !important;
}}
textarea::placeholder, input::placeholder {{
    color: #aeaeb2 !important;
}}

/* 7 ─ Labels */
label > span:first-of-type,
.label-wrap > span,
.dark label > span:first-of-type {{
    font-size: 13px !important;
    font-weight: 600 !important;
    color: {DARK} !important;
    letter-spacing: 0.05px !important;
    text-transform: none !important;
    display: block !important;
    margin-bottom: 6px !important;
}}

/* 8 ─ Slider: blue accent */
input[type=range] {{ accent-color: {BLUE} !important; }}
.dark input[type=range] {{ accent-color: {BLUE} !important; }}

/* 9 ─ Primary CTA */
button.primary, .dark button.primary {{
    background: {BLUE} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 980px !important;
    padding: 13px 32px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: -0.1px !important;
    width: 100% !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
    box-shadow: 0 1px 5px rgba(0,113,227,0.28) !important;
}}
button.primary:hover {{ background: #0077ed !important; transform: translateY(-1px) !important; }}
button.primary:active {{ background: #006edb !important; transform: none !important; }}

/* 10 ─ Group card: white card for input section */
.input-card > div {{
    background: {WHITE} !important;
    border-radius: 18px !important;
    padding: 28px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07) !important;
    border: none !important;
}}
.dark .input-card > div {{
    background: {WHITE} !important;
}}

/* 11 ─ Output HTML areas: transparent */
.output-html {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _match_color(pct: int) -> str:
    if pct >= 75: return "#34c759"
    if pct >= 50: return "#ff9f0a"
    return "#ff453a"


def _section_label(text: str) -> str:
    return (
        f"<p style='margin:0 0 14px;font-size:11px;font-weight:600;"
        f"color:{LIGHT};letter-spacing:0.7px;text-transform:uppercase;"
        f"font-family:{FONT};'>{text}</p>"
    )


def _story_card(story: dict, is_recommended: bool = False) -> str:
    # distance is cosine distance: 0 = identical, 2 = opposite.
    # cosine_similarity = 1 - distance, so match % = (1 - distance) * 100.
    # Clamp to [0, 100] — values < 0 would mean cosine_sim < 0 (essentially unrelated).
    pct    = max(0, min(100, int((1 - story["distance"]) * 100)))
    col    = _match_color(pct)
    title  = story["metadata"]["title"]
    themes = story["metadata"]["themes"]
    body   = story["document"]
    accent = f"border-left:3px solid {BLUE};" if is_recommended else ""

    badge = ""
    if is_recommended:
        badge = (
            f"<div style='margin-bottom:12px;'>"
            f"<span style='display:inline-block;background:{BLUE};color:#fff;"
            f"font-size:10.5px;font-weight:700;padding:3px 11px;border-radius:20px;"
            f"letter-spacing:0.8px;text-transform:uppercase;font-family:{FONT};'>"
            f"⭐ &nbsp;Recommended</span></div>"
        )

    return f"""
<div style="background:{WHITE};border-radius:16px;padding:22px 24px;margin-bottom:14px;
            box-shadow:0 2px 12px rgba(0,0,0,0.06),0 1px 3px rgba(0,0,0,0.04);{accent}">
  {badge}
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:14px;margin-bottom:8px;">
    <h3 style="margin:0;font-size:16px;font-weight:700;color:{DARK};line-height:1.3;font-family:{FONT};">{title}</h3>
    <span style="flex-shrink:0;background:{col}1a;color:{col};font-size:12px;font-weight:700;
                 padding:4px 12px;border-radius:20px;white-space:nowrap;font-family:{FONT};">{pct}% match</span>
  </div>
  <p style="margin:0 0 12px;font-size:11px;color:{LIGHT};font-weight:500;letter-spacing:0.5px;
            text-transform:uppercase;font-family:{FONT};">{themes}</p>
  <p style="margin:0;font-size:14px;color:#3d3d3f;line-height:1.72;font-family:{FONT};">{body}</p>
</div>"""


# ── Core callback ──────────────────────────────────────────────────────────────
def coach(question: str, n_results: int):
    if not question.strip():
        empty = (
            f"<p style='color:{LIGHT};font-family:{FONT};text-align:center;"
            f"padding:52px 0;font-size:15px;'>Results will appear here.</p>"
        )
        return empty, empty, empty

    result  = answer_question(question, n_results=int(n_results))
    stories = sorted(result["retrieved_stories"], key=lambda s: s["distance"])

    recommended = stories[0]
    others      = stories[1:]

    # ── 1. Polished story card ────────────────────────────────────────────────
    polished_body = (
        result["polished_story"]
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n\n", f"</p><p style='margin:0 0 14px;font-size:15px;color:{DARK};line-height:1.8;font-family:{FONT};'>")
        .replace("\n", "<br>")
    )
    polished_html = f"""
<div style="background:{WHITE};border-radius:16px;padding:28px 30px;margin-bottom:20px;
            box-shadow:0 4px 20px rgba(0,0,0,0.08),0 1px 3px rgba(0,0,0,0.04);
            border-left:3px solid #34c759;">
  {_section_label("✨ &nbsp;Your Polished Story — Ready to Practice")}
  <p style="margin:0 0 14px;font-size:15px;color:{DARK};line-height:1.8;font-family:{FONT};">{polished_body}</p>
</div>"""

    # ── 2. Coaching card ──────────────────────────────────────────────────────
    coaching_body = (
        result["recommendation"]
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n\n", f"</p><p style='margin:0 0 10px;font-size:14.5px;color:{DARK};line-height:1.75;font-family:{FONT};'>")
        .replace("\n", "<br>")
    )
    coaching_html = f"""
<div style="background:{WHITE};border-radius:16px;padding:24px 26px;margin-bottom:20px;
            box-shadow:0 2px 12px rgba(0,0,0,0.06),0 1px 3px rgba(0,0,0,0.04);">
  {_section_label("🤖 &nbsp;Coaching Analysis")}
  <p style="margin:0;font-size:14.5px;color:{DARK};line-height:1.75;font-family:{FONT};">{coaching_body}</p>
</div>"""

    # ── 3. Stories: recommended first, others by match score ──────────────────
    cards = _story_card(recommended, is_recommended=True)
    for s in others:
        cards += _story_card(s)
    stories_html = f"<div>{_section_label('📚 &nbsp;Matched Stories')}{cards}</div>"

    return polished_html, coaching_html, stories_html


# ── App layout ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Interview Coach", theme=theme, css=APPLE_CSS) as app:

    # Hero
    gr.HTML(f"""
    <div style="text-align:center;margin-bottom:40px;padding-top:8px;">
      <div style="font-size:38px;font-weight:700;color:{DARK};
                  letter-spacing:-0.8px;line-height:1.1;margin-bottom:10px;
                  font-family:{FONT};">🎯 Interview Coach</div>
      <div style="font-size:17px;color:{MID};font-weight:400;font-family:{FONT};">
        Paste an interview question — get a polished story ready to practice.
      </div>
    </div>
    """)

    # Input card
    with gr.Group(elem_classes="input-card"):
        question = gr.Textbox(
            label="Interview Question",
            placeholder="e.g. Tell me about a time you led a cross-functional initiative...",
            lines=3,
        )
        n_results = gr.Slider(
            minimum=1, maximum=5, value=3, step=1,
            label="Number of stories to retrieve",
        )
        submit = gr.Button("Find My Best Stories →", variant="primary")

    gr.HTML(f"<div style='height:24px;'></div>")

    # Results: polished story → coaching → stories
    polished_out = gr.HTML()
    coaching_out = gr.HTML()
    stories_out  = gr.HTML()

    submit.click(
        fn=coach,
        inputs=[question, n_results],
        outputs=[polished_out, coaching_out, stories_out],
    )

app.launch()
