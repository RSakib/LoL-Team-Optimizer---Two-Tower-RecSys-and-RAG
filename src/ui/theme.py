"""League-inspired presentation. The supplied Riot font is scoped to the hero title."""
import base64
from pathlib import Path

import gradio as gr

TITLE = "League of Legends: Team Recommender"
FONT_PATH = Path(__file__).resolve().parents[2] / "assets/fonts/BeaufortforLOL-Bold.ttf"


def build_theme():
    colors = {
        "body_background_fill": "#010a13", "body_text_color": "#f0e6d2",
        "body_text_color_subdued": "#a3b4bc", "background_fill_primary": "#071622",
        "background_fill_secondary": "#0c202c", "border_color_primary": "#28424b",
        "border_color_accent": "#c8aa6e", "border_color_accent_subdued": "#6b5833",
        "color_accent_soft": "#12343d", "link_text_color": "#62d3dc",
        "block_background_fill": "#0a1925", "block_border_color": "#28424b",
        "block_label_text_color": "#f0e6d2", "block_title_text_color": "#f0e6d2",
        "block_info_text_color": "#a3b4bc", "panel_background_fill": "#0a1925",
        "panel_border_color": "#28424b", "input_background_fill": "#06121c",
        "input_background_fill_focus": "#0b202b", "input_background_fill_hover": "#0b202b",
        "input_border_color": "#37525a", "input_border_color_focus": "#0ac8b9",
        "input_border_color_hover": "#c8aa6e", "input_placeholder_color": "#8ca0aa",
        "button_primary_background_fill": "linear-gradient(110deg, #c8aa6e, #dfc48a)",
        "button_primary_background_fill_hover": "linear-gradient(110deg, #dfc48a, #f0e6d2)",
        "button_primary_border_color": "#f0e6d2", "button_primary_border_color_hover": "#f0e6d2",
        "button_primary_text_color": "#07131c", "button_primary_text_color_hover": "#07131c",
        "button_secondary_background_fill": "#0c2632", "button_secondary_background_fill_hover": "#123c47",
        "button_secondary_border_color": "#3d747b", "button_secondary_border_color_hover": "#0ac8b9",
        "button_secondary_text_color": "#cdfafa", "button_secondary_text_color_hover": "#ffffff",
        "slider_color": "#0ac8b9", "loader_color": "#c8aa6e", "accordion_text_color": "#c8aa6e",
    }
    # The same palette in both browser modes prevents white inputs in a dark page.
    values = {key: value for name, value in colors.items() for key in (name, name + "_dark")}
    return gr.themes.Base(font=["Arial", "sans-serif"]).set(
        **values, color_accent="#0ac8b9", block_radius="6px", input_radius="4px",
        button_large_radius="4px", button_small_radius="4px", button_medium_radius="4px",
        block_label_text_size="13px", block_label_text_weight="600", layout_gap="18px",
    )


THEME = build_theme()
FONT_CSS = (
    "@font-face { font-family: 'Beaufort for LoL'; font-style: normal; font-weight: 700; "
    "font-display: swap; src: url('data:font/ttf;base64,"
    + base64.b64encode(FONT_PATH.read_bytes()).decode("ascii") + "') format('truetype'); }"
)

FONT_HEAD = '<style id="league-title-font">' + FONT_CSS + '</style>'

GLOBAL_CSS = """
body { background: #010a13; }
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 28px 32px 20px !important;
    background: radial-gradient(ellipse at 90% 0%, #07323c80, transparent 46%), #010a13 !important;
}
#finder-panel {
    border: 1px solid #28424b; border-top: 2px solid #8b7443;
    border-radius: 6px; padding: 22px; background: #0a1925;
    box-shadow: 0 12px 36px #00000030;
}
#team-preference textarea { line-height: 1.6; }
#build-team { min-height: 54px; letter-spacing: .08em; font-weight: 700; }
.gradio-container button:focus-visible, .gradio-container summary:focus-visible {
    outline: 2px solid #0ac8b9 !important; outline-offset: 3px;
}
#lineup-scout-panel { border-top: 1px solid #6b5833; padding-top: 12px; }
#lineup-scout-report .prose { line-height: 1.7; }
#project-note { color: #a3b4bc; font-size: 12px; padding: 8px 0; }
@media (max-width: 640px) {
    .gradio-container { padding: 16px 14px !important; }
    #finder-panel { padding: 14px; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
"""

HERO_HTML = """
<header class="league-hero">
  <a class="hero-writeup" href="https://github.com/RSakib/LoL-Team-Optimizer---Two-Tower-RecSys-and-RAG" target="_blank" rel="noopener noreferrer" title="Read the Write-up">
    <span class="writeup-icon" aria-hidden="true">?</span><span>Read the Write-up</span>
  </a>
  <div class="hero-copy">
    <h1 id="league-title">League of Legends: <span>Team Recommender</span></h1>
    <p class="hero-description">Match real players to your roles, rank, and playstyle preferences.</p>
    <div class="hero-tags"><span>REAL MATCH DATA</span><span>RAG SCOUT REPORTS</span></div>
  </div>
  <div class="hero-crest" aria-hidden="true"><div class="crest-ring"><span>04</span><small>OPEN ROLES</small></div></div>
</header>
"""

HERO_CSS = """
.league-hero { position:relative; display:flex; align-items:center; justify-content:space-between; gap:24px; padding:46px 0 30px; border-bottom:1px solid #6b5833; margin-bottom:8px; }
.hero-writeup { position:absolute; top:0; right:0; display:inline-flex; align-items:center; gap:8px; min-height:36px; padding:2px 4px; color:#c8aa6e; font:600 12px/1.5 Arial,sans-serif; text-decoration:none; border-radius:4px; }
.hero-writeup:hover { color:#f0e6d2; text-decoration:underline; }
.hero-writeup:focus-visible { outline:2px solid #0ac8b9; outline-offset:3px; }
.writeup-icon { display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; border:1px solid currentColor; border-radius:50%; font-size:15px; }
.hero-copy { min-width:0; }
#league-title { font-family:'Beaufort for LoL',Georgia,serif; font-weight:700; font-size:clamp(30px,4.3vw,54px); line-height:1.08; letter-spacing:.02em; color:#c8aa6e; margin:20px 0 14px; }
#league-title span { display:block; color:#f0e6d2; }
.hero-description { max-width:640px; color:#a3b4bc; font:400 14px/1.7 Arial,sans-serif; margin:0 0 20px; }
.hero-tags { display:flex; flex-wrap:wrap; gap:8px; }
.hero-tags span { font:600 9px/1.5 Arial,sans-serif; letter-spacing:.1em; padding:5px 9px; border:1px solid #28424b; color:#a3b4bc; background:#081c27; }
.hero-crest { padding:20px 30px; flex:0 0 auto; }
.crest-ring { width:146px; height:146px; border:1px solid #8b7443; border-radius:50%; position:relative; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:0 0 60px #0ac8b912; }
.crest-ring:before { content:''; position:absolute; inset:15px; border:1px solid #235865; transform:rotate(45deg); }
.crest-ring:after { content:''; position:absolute; inset:-9px; border:1px solid #28424b; border-radius:50%; }
.crest-ring span { color:#c8aa6e; font:300 42px/1 Arial,sans-serif; position:relative; }
.crest-ring small { color:#7dd9db; font:600 8px/1.5 Arial,sans-serif; letter-spacing:.16em; margin-top:10px; position:relative; }
@media(max-width:700px) { .hero-crest { display:none; } .league-hero { padding-bottom:22px; } }
"""

SECTION_CSS = """
.section-heading { display:flex; gap:12px; align-items:center; color:#f0e6d2; font:600 13px/1.5 Arial,sans-serif; letter-spacing:.1em; text-transform:uppercase; margin:5px 0 0; }
.section-heading span { color:#c8aa6e; font-size:11px; border:1px solid #6b5833; padding:4px 6px; }
"""

EMPTY_HTML = """
<section class="team-results empty-lineup">
 <div class="empty-mark" aria-hidden="true">◇</div>
 <h2>Your lineup starts here</h2>
 <p>Choose your role and preferences, then build a team from recorded player history.</p>
 <div class="empty-slots" aria-hidden="true"><span>01</span><span>02</span><span>03</span><span>04</span></div>
</section>
"""
