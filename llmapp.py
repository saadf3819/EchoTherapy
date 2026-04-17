import streamlit as st
import numpy as np
import torch
import librosa
import tempfile
import os
import random
import anthropic
from pathlib import Path
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Config,
)
import plotly.graph_objects as go

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Echo Therapy · Emotion Detector",
    page_icon="🎙️",
    layout="centered",
)

# ── Emotion mapping ────────────────────────────────────────────────────────────
EMOTION_MAP = {
    "LABEL_0": {"name": "Neutral",  "emoji": "😐", "color": "#94a3b8"},
    "LABEL_1": {"name": "Happy",    "emoji": "😄", "color": "#facc15"},
    "LABEL_2": {"name": "Sad",      "emoji": "😢", "color": "#60a5fa"},
    "LABEL_3": {"name": "Angry",    "emoji": "😠", "color": "#f87171"},
    "LABEL_4": {"name": "Fearful",  "emoji": "😨", "color": "#a78bfa"},
}

# ── Music Recommendations ──────────────────────────────────────────────────────
# Each emotion has curated YouTube Music / Spotify embed URLs and metadata.
# The embed URLs use YouTube's nocookie domain for privacy-friendly embedding.
# Replace playlist IDs with your own curated playlists as needed.
MUSIC_RECOMMENDATIONS = {
    "LABEL_0": {  # Neutral
        "mood_label": "Focused & Balanced",
        "mood_desc": "Lo-fi beats and ambient soundscapes to keep you grounded.",
        "gradient": "linear-gradient(135deg, #1e293b, #0f172a)",
        "accent": "#94a3b8",
        "tracks": [
            {
                "title": "Lofi Hip Hop Radio – Beats to Relax/Study",
                "artist": "Lofi Girl",
                "genre": "Lo-Fi / Chill",
                "embed_url": "https://www.youtube-nocookie.com/embed/jfKfPfyJRdk",
                "thumbnail": "🎵",
            },
            {
                "title": "Ambient Study Music To Concentrate",
                "artist": "Greenred Productions",
                "genre": "Ambient / Focus",
                "embed_url": "https://www.youtube-nocookie.com/embed/sjkrrmBnpGE",
                "thumbnail": "🎶",
            },
            {
                "title": "Peaceful Piano – Relaxing Music",
                "artist": "Soothing Relaxation",
                "genre": "Classical / Piano",
                "embed_url": "https://www.youtube-nocookie.com/embed/77ZozI0rw7w",
                "thumbnail": "🎹",
            },
        ],
    },
    "LABEL_1": {  # Happy
        "mood_label": "Euphoric & Energised",
        "mood_desc": "Feel-good anthems and upbeat grooves to ride your high.",
        "gradient": "linear-gradient(135deg, #451a03, #1c1007)",
        "accent": "#facc15",
        "tracks": [
            {
                "title": "Happy – Official Video",
                "artist": "Pharrell Williams",
                "genre": "Pop / Soul",
                "embed_url": "https://www.youtube-nocookie.com/embed/ZbZSe6N_BXs",
                "thumbnail": "😄",
            },
            {
                "title": "Good as Hell",
                "artist": "Lizzo",
                "genre": "Pop / R&B",
                "embed_url": "https://www.youtube-nocookie.com/embed/SmbmeOgGsKc",
                "thumbnail": "🌟",
            },
            {
                "title": "Can't Stop the Feeling!",
                "artist": "Justin Timberlake",
                "genre": "Pop / Funk",
                "embed_url": "https://www.youtube-nocookie.com/embed/ru0K8uYEZWw",
                "thumbnail": "🕺",
            },
        ],
    },
    "LABEL_2": {  # Sad
        "mood_label": "Reflective & Tender",
        "mood_desc": "Gentle melodies that honour your feelings and hold space.",
        "gradient": "linear-gradient(135deg, #0c1a2e, #060d18)",
        "accent": "#60a5fa",
        "tracks": [
            {
                "title": "The Night We Met",
                "artist": "Lord Huron",
                "genre": "Indie Folk",
                "embed_url": "https://www.youtube-nocookie.com/embed/KtlgYxa6BMU",
                "thumbnail": "🌙",
            },
            {
                "title": "Skinny Love",
                "artist": "Bon Iver",
                "genre": "Folk / Indie",
                "embed_url": "https://www.youtube-nocookie.com/embed/scdclYHCKPc",
                "thumbnail": "🍂",
            },
            {
                "title": "Someone Like You",
                "artist": "Adele",
                "genre": "Soul / Pop",
                "embed_url": "https://www.youtube-nocookie.com/embed/hLQl3WQQoQ0",
                "thumbnail": "💧",
            },
        ],
    },
    "LABEL_3": {  # Angry
        "mood_label": "Release & Catharsis",
        "mood_desc": "High-energy tracks to channel and release that tension.",
        "gradient": "linear-gradient(135deg, #2d0a0a, #1a0404)",
        "accent": "#f87171",
        "tracks": [
            {
                "title": "Break Stuff",
                "artist": "Limp Bizkit",
                "genre": "Nu-Metal / Rock",
                "embed_url": "https://www.youtube-nocookie.com/embed/ZpUYjpKg9KY",
                "thumbnail": "🔥",
            },
            {
                "title": "Given Up",
                "artist": "Linkin Park",
                "genre": "Rock / Alternative",
                "embed_url": "https://www.youtube-nocookie.com/embed/0xyxtzD54eM",
                "thumbnail": "⚡",
            },
            {
                "title": "Killing in the Name",
                "artist": "Rage Against the Machine",
                "genre": "Metal / Rock",
                "embed_url": "https://www.youtube-nocookie.com/embed/bWXazVhlyxQ",
                "thumbnail": "💥",
            },
        ],
    },
    "LABEL_4": {  # Fearful
        "mood_label": "Calm & Reassuring",
        "mood_desc": "Soothing soundscapes and gentle harmonies to ease anxiety.",
        "gradient": "linear-gradient(135deg, #1a0a2e, #0d0617)",
        "accent": "#a78bfa",
        "tracks": [
            {
                "title": "Weightless",
                "artist": "Marconi Union",
                "genre": "Ambient / Therapy",
                "embed_url": "https://www.youtube-nocookie.com/embed/UfcAVejslrU",
                "thumbnail": "🌊",
            },
            {
                "title": "Claire de Lune",
                "artist": "Claude Debussy",
                "genre": "Classical / Piano",
                "embed_url": "https://www.youtube-nocookie.com/embed/WNcsUNKnbDY",
                "thumbnail": "🌸",
            },
            {
                "title": "Pure Shores",
                "artist": "All Saints",
                "genre": "Pop / Ambient",
                "embed_url": "https://www.youtube-nocookie.com/embed/7HBRgqMv4IA",
                "thumbnail": "🏝️",
            },
        ],
    },
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp                      { background: #0d0f14; color: #e2e8f0; }
.main .block-container      { padding-top: 2rem; max-width: 720px; }
.hero                       { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero h1 {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 3rem;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.hero p {
    color: #64748b; font-family: 'Space Mono', monospace;
    font-size: 0.85rem; letter-spacing: 0.05em;
}
.result-box {
    background: linear-gradient(135deg, #1e1b4b, #0f172a);
    border: 1px solid #312e81; border-radius: 20px;
    padding: 2rem; text-align: center; margin: 1.5rem 0;
}
.result-emoji { font-size: 5rem; line-height: 1; }
.result-label {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 2.5rem; margin: 0.5rem 0 0.25rem;
}
.result-conf {
    font-family: 'Space Mono', monospace; color: #64748b;
    font-size: 0.8rem; letter-spacing: 0.1em;
}
.section-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.15em; color: #475569;
    text-transform: uppercase; margin-bottom: 0.75rem;
}

/* ── Music Recommendation Card ── */
.music-section {
    margin: 2rem 0 0.5rem;
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}
.music-header {
    padding: 1.5rem 1.75rem 1rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}
.music-icon { font-size: 2.2rem; line-height: 1; }
.music-header-text {}
.music-mood-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 0.2rem;
    opacity: 0.6;
}
.music-mood-title {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 1.35rem; margin-bottom: 0.1rem;
}
.music-mood-desc {
    font-family: 'Syne', sans-serif; font-size: 0.82rem;
    opacity: 0.55; line-height: 1.5;
}
.track-list { padding: 0 1.25rem 1.25rem; display: flex; flex-direction: column; gap: 0.6rem; }
.track-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 1rem 1.2rem;
    display: flex; align-items: center; gap: 1rem;
    cursor: pointer; transition: background 0.2s, border-color 0.2s;
}
.track-card:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.14);
}
.track-thumb {
    font-size: 1.8rem; width: 44px; height: 44px;
    border-radius: 10px; background: rgba(255,255,255,0.06);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.track-meta { flex: 1; min-width: 0; }
.track-title {
    font-family: 'Syne', sans-serif; font-weight: 600;
    font-size: 0.92rem; margin-bottom: 0.15rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.track-artist {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    opacity: 0.5;
}
.track-genre {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    padding: 0.15rem 0.5rem; border-radius: 999px;
    background: rgba(255,255,255,0.06); opacity: 0.7;
    white-space: nowrap; flex-shrink: 0;
}
.track-play-btn {
    font-size: 1.2rem; opacity: 0.5; flex-shrink: 0;
    transition: opacity 0.2s;
}
.track-card:hover .track-play-btn { opacity: 1; }

/* Embed player */
.player-wrap {
    margin: 0.5rem 1.25rem 1.25rem;
    border-radius: 14px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.player-wrap iframe {
    display: block; width: 100%; border: none;
}

.badge {
    display: inline-block; padding: 0.2rem 0.6rem;
    border-radius: 999px; font-family: 'Space Mono', monospace; font-size: 0.7rem;
}
.badge-green { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-red   { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }

/* ── AI Companion Card ── */
.ai-companion-wrap {
    margin: 2rem 0 0.5rem;
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(135deg, #0f1e2e, #0a1520);
    padding: 1.5rem 1.75rem;
    position: relative;
}
.ai-companion-wrap::before {
    content: '';
    position: absolute; inset: 0;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(167,139,250,0.06), rgba(96,165,250,0.06));
    pointer-events: none;
}
.ai-header {
    display: flex; align-items: center; gap: 0.75rem;
    margin-bottom: 1rem;
}
.ai-avatar {
    width: 42px; height: 42px; border-radius: 50%;
    background: linear-gradient(135deg, #6d28d9, #2563eb);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
    box-shadow: 0 0 18px rgba(109,40,217,0.45);
}
.ai-header-text {}
.ai-header-name {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 0.95rem; color: #e2e8f0;
}
.ai-header-sub {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.15em; color: #475569;
    text-transform: uppercase;
}
.ai-bubble {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px 18px 18px 18px;
    padding: 1.1rem 1.3rem;
    font-family: 'Syne', sans-serif; font-size: 0.93rem;
    line-height: 1.75; color: #cbd5e1;
    white-space: pre-wrap;
}
.ai-typing {
    display: flex; gap: 5px; align-items: center; padding: 0.5rem 0;
}
.ai-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #6d28d9; opacity: 0.7;
    animation: ai-bounce 1.2s infinite;
}
.ai-dot:nth-child(2) { animation-delay: 0.2s; }
.ai-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes ai-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.7; }
    40% { transform: translateY(-6px); opacity: 1; }
}
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #4f46e5);
    color: white; border: none; border-radius: 10px;
    font-family: 'Syne', sans-serif; font-weight: 600;
    padding: 0.6rem 2rem; font-size: 1rem; width: 100%; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
audio { width: 100%; border-radius: 8px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎙️ Echo Theapy</h1>
  <p>WAV2VEC2 · SPEECH EMOTION RECOGNITION · MUSIC THERAPY</p>
</div>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR     = Path(".")
MODEL_WEIGHTS = MODEL_DIR / "model.emotion"

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_dir_str, weights_path_str):
    import traceback
    try:
        model_dir    = Path(model_dir_str)
        weights_path = Path(weights_path_str)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weight file not found: {weights_path.resolve()}\n"
                "Make sure model.emotion is in the same folder as app.py."
            )

        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir))
        config    = Wav2Vec2Config.from_pretrained(str(model_dir))
        model     = Wav2Vec2ForSequenceClassification(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(str(weights_path), "rb") as f:
            first_bytes = f.read(8)

        is_safetensors = not (
            first_bytes[:2] in (b"\x80\x02", b"\x80\x04", b"\x80\x05")
            or first_bytes[:2] == b"PK"
        )

        if is_safetensors:
            from safetensors.torch import load_file
            checkpoint = load_file(str(weights_path), device=str(device))
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        else:
            checkpoint = torch.load(str(weights_path), map_location=device)
            if isinstance(checkpoint, dict):
                for wrap_key in ("model", "model_state_dict", "state_dict"):
                    if wrap_key in checkpoint and isinstance(checkpoint[wrap_key], dict):
                        checkpoint = checkpoint[wrap_key]
                        break
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)

        if missing:
            print(f"[warn] missing keys: {missing[:5]}")
        if unexpected:
            print(f"[warn] unexpected keys: {unexpected[:5]}")

        model.to(device)
        model.eval()
        return processor, model, device, None

    except Exception:
        return None, None, None, traceback.format_exc()


# ── Status badges ──────────────────────────────────────────────────────────────
processor = model = device = load_error = None

def ensure_model_loaded():
    global processor, model, device, load_error
    if model is not None or load_error is not None:
        return

    import threading, time

    result = {}
    def _load():
        result["out"] = load_model(str(MODEL_DIR), str(MODEL_WEIGHTS))

    t = threading.Thread(target=_load, daemon=True)
    t.start()

    placeholder = st.empty()
    dots = 0
    while t.is_alive():
        dots = (dots % 3) + 1
        placeholder.info(f"Loading model.emotion{'.' * dots}  (large file, please wait)")
        import time as _time; _time.sleep(0.6)
    placeholder.empty()

    processor, model, device, load_error = result["out"]


c1, c2 = st.columns(2)
with c1:
    if MODEL_WEIGHTS.exists():
        st.markdown('<span class="badge badge-green">● model.emotion found</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">⚠ model.emotion not found</span>',
                    unsafe_allow_html=True)
with c2:
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    st.markdown(f'<span class="badge badge-green">● Device: {dev_str.upper()}</span>',
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload Audio File</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Supported: WAV, MP3, FLAC, OGG, M4A",
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def predict_emotion(audio_path):
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    waveform, _ = librosa.effects.trim(waveform, top_db=20)

    inputs = processor(
        waveform, sampling_rate=16000,
        return_tensors="pt", padding=True,
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id    = int(np.argmax(probs))
    label_key  = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])
    all_scores = {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}
    return label_key, confidence, all_scores, waveform


def bar_chart(scores):
    labels = [EMOTION_MAP.get(k, {"name": k})["name"]    for k in scores]
    emojis = [EMOTION_MAP.get(k, {"emoji": ""})["emoji"] for k in scores]
    values = list(scores.values())
    colors = [EMOTION_MAP.get(k, {"color": "#6d28d9"})["color"] for k in scores]
    fig = go.Figure(go.Bar(
        x=values,
        y=[f"{e} {l}" for e, l in zip(emojis, labels)],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="Space Mono", color="#94a3b8", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(family="Syne", size=14, color="#e2e8f0")),
        margin=dict(l=10, r=60, t=10, b=10), height=240,
    )
    return fig


def waveform_chart(wav):
    sr   = 16000
    step = max(1, len(wav) // 800)
    s, t = wav[::step], np.linspace(0, len(wav) / sr, len(wav[::step]))
    fig  = go.Figure(go.Scatter(
        x=t, y=s, mode="lines",
        line=dict(color="#6d28d9", width=1),
        fill="tozeroy", fillcolor="rgba(109,40,217,0.15)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title="seconds",
                   tickfont=dict(family="Space Mono", color="#475569", size=10)),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=10, r=10, t=10, b=30), height=130,
    )
    return fig


AI_SYSTEM_PROMPT = """You are an emotionally intelligent AI assistant integrated into a voice-based emotion detection application.
Your role is to:
1. Understand the user's emotional state based on the detected emotion label and conversation context.
2. Respond in a natural, human-like, and empathetic tone.
3. Provide helpful, safe, and practical suggestions (remedies) tailored to the user's emotional condition.

Guidelines:
* Always acknowledge the user's emotion first (e.g., "It sounds like you're feeling stressed").
* Be supportive, calm, and non-judgmental.
* Keep responses concise and meaningful, suitable for voice interaction.
* Use simple, clear, and conversational language.
* Suggest 2–4 practical remedies (e.g., breathing exercises, taking a short break, talking to a friend, journaling).
* Avoid giving medical, clinical, or diagnostic advice.
* Do not sound robotic or overly formal.

Emotion Handling:
* If emotion is "Neutral": keep the conversation friendly and engaging.
* If emotion is "Happy": reinforce positivity and encourage maintaining the mood.
* If emotion is "Sad": provide emotional support and gentle uplifting suggestions.
* If emotion is "Angry": encourage pausing, breathing, and reflecting before reacting.
* If emotion is "Fearful": suggest calming techniques such as deep breathing, grounding, or a short walk.

Safety:
* If the user expresses extreme distress, panic, or harmful thoughts, respond with care and encourage reaching out to a trusted person or a professional.

Output Style:
* Conversational, warm and human-like.
* Short sentences suitable for voice playback.
* Use line breaks between thoughts for readability.
Always prioritize the user's emotional well-being."""


@st.cache_data(show_spinner=False)
def get_ai_response(emotion_name: str) -> str:
    """Call the Anthropic API to get an empathetic response for the detected emotion."""
    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=AI_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"The voice analysis has detected that I'm feeling: {emotion_name}. "
                        f"Please respond to me with empathy and give me helpful suggestions."
                    ),
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        return f"I'm having a little trouble connecting right now. But I can see you're feeling {emotion_name} — please be gentle with yourself. 💙"


def render_ai_assistant(emotion_name: str, accent_color: str):
    """Render the AI emotional companion panel."""
    st.markdown('<div class="section-label">🤖 AI Emotional Companion</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-companion-wrap">
        <div class="ai-header">
            <div class="ai-avatar">🧠</div>
            <div class="ai-header-text">
                <div class="ai-header-name">Emo · Your Emotional Companion</div>
                <div class="ai-header-sub">Powered by Claude · Empathetic AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Emo is thinking…"):
        response_text = get_ai_response(emotion_name)

    st.markdown(f"""
    <div class="ai-companion-wrap" style="margin-top:0.5rem; border-top: none; border-radius: 0 0 24px 24px; padding-top: 0.5rem;">
        <div class="ai-bubble">{response_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_music_recommendations(label_key: str, emotion_name: str, accent_color: str):
    """
    Renders the contextual music recommendation panel.
    Users can browse curated tracks and click to load an embedded YouTube player.
    """
    rec = MUSIC_RECOMMENDATIONS.get(label_key)
    if not rec:
        return  # No recommendations for this label

    tracks = rec["tracks"]
    mood_icon = EMOTION_MAP.get(label_key, {}).get("emoji", "🎵")

    # ── Section header
    st.markdown('<div class="section-label">🎵 Contextual Music Recommendation</div>',
                unsafe_allow_html=True)

    # ── Card wrapper with emotion-specific gradient
    st.markdown(f"""
    <div class="music-section" style="background: {rec['gradient']};">
        <div class="music-header">
            <div class="music-icon">{mood_icon}</div>
            <div class="music-header-text">
                <div class="music-mood-label" style="color:{rec['accent']}">
                    Detected · {emotion_name.upper()}
                </div>
                <div class="music-mood-title" style="color:{rec['accent']}">
                    {rec['mood_label']}
                </div>
                <div class="music-mood-desc">{rec['mood_desc']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state for which track is selected
    session_key = f"selected_track_{label_key}"
    if session_key not in st.session_state:
        st.session_state[session_key] = 0  # default: first track

    # ── Track selection buttons (rendered via Streamlit columns for interactivity)
    st.markdown(f"""
    <div style="
        background: {rec['gradient']};
        border: 1px solid rgba(255,255,255,0.07);
        border-top: none;
        border-radius: 0 0 24px 24px;
        padding: 0 1.25rem 0.25rem;
    ">
    """, unsafe_allow_html=True)

    for i, track in enumerate(tracks):
        is_selected = (st.session_state[session_key] == i)
        selected_style = f"border-color: {rec['accent']}44; background: rgba(255,255,255,0.08);" if is_selected else ""

        col_track, col_btn = st.columns([5, 1])
        with col_track:
            st.markdown(f"""
            <div class="track-card" style="{selected_style}">
                <div class="track-thumb">{track['thumbnail']}</div>
                <div class="track-meta">
                    <div class="track-title">{track['title']}</div>
                    <div class="track-artist">{track['artist']}</div>
                </div>
                <div class="track-genre">{track['genre']}</div>
                <div class="track-play-btn">{'▶️' if not is_selected else '🔊'}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            btn_label = "Now Playing" if is_selected else f"Play"
            if st.button(btn_label, key=f"play_{label_key}_{i}", disabled=is_selected):
                st.session_state[session_key] = i
                st.rerun()

    # ── Embedded YouTube player for selected track
    selected_track = tracks[st.session_state[session_key]]
    st.markdown(f"""
    <div class="player-wrap" style="margin: 0.75rem 0 0.5rem;">
        <iframe
            src="{selected_track['embed_url']}?autoplay=1&rel=0&modestbranding=1"
            height="200"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            title="{selected_track['title']} by {selected_track['artist']}"
        ></iframe>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Shuffle option
    col_shuf, _ = st.columns([2, 3])
    with col_shuf:
        if st.button("🔀 Shuffle Track", key=f"shuffle_{label_key}"):
            current = st.session_state[session_key]
            others  = [i for i in range(len(tracks)) if i != current]
            st.session_state[session_key] = random.choice(others)
            st.rerun()


# ── Main flow ──────────────────────────────────────────────────────────────────
if uploaded:
    st.audio(uploaded, format=uploaded.type)
    run = st.button("Detect Emotion")

    if run:
        ensure_model_loaded()
        if load_error:
            st.error("Failed to load model.")
            with st.expander("Show error"):
                st.code(load_error)
            st.stop()

        suffix = Path(uploaded.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Analysing emotion…"):
                label_key, confidence, all_scores, waveform = predict_emotion(tmp_path)

            emotion = EMOTION_MAP.get(
                label_key, {"name": label_key, "emoji": "🎙️", "color": "#6d28d9"}
            )

            # ── Emotion result card
            st.markdown(f"""
            <div class="result-box">
              <div class="result-emoji">{emotion["emoji"]}</div>
              <div class="result-label" style="color:{emotion["color"]}">{emotion["name"]}</div>
              <div class="result-conf">CONFIDENCE &middot; {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Probabilities chart
            st.markdown('<div class="section-label">Emotion Probabilities</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(bar_chart(all_scores), use_container_width=True,
                            config={"displayModeBar": False})

            # ── Waveform chart
            st.markdown('<div class="section-label">Audio Waveform</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(waveform_chart(waveform), use_container_width=True,
                            config={"displayModeBar": False})

            st.divider()

            # ── 🤖 AI Emotional Companion
            render_ai_assistant(
                emotion_name=emotion["name"],
                accent_color=emotion["color"],
            )

            st.divider()

            # ── 🎵 Contextual Music Recommendation (NEW FEATURE)
            render_music_recommendations(
                label_key=label_key,
                emotion_name=emotion["name"],
                accent_color=emotion["color"],
            )

            # ── Raw scores
            with st.expander("Raw Scores"):
                for k, v in sorted(all_scores.items(), key=lambda x: -x[1]):
                    em = EMOTION_MAP.get(k, {})
                    st.markdown(f"`{k}` · **{em.get('name', k)}** {em.get('emoji','')} — `{v*100:.2f}%`")

        except Exception as e:
            import traceback
            st.error(f"Prediction failed: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
        finally:
            os.unlink(tmp_path)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Emotion Labels")
    st.caption("Rename labels to match your dataset.")
    for key in EMOTION_MAP:
        EMOTION_MAP[key]["name"] = st.text_input(
            key, value=EMOTION_MAP[key]["name"], key=f"label_{key}"
        )

    st.divider()
    st.markdown("### 🎵 Music Recommendations")
    st.caption("Each emotion maps to 3 curated tracks. Edit `MUSIC_RECOMMENDATIONS` in the source to swap in your own YouTube embeds or Spotify iframes.")
    for label_key, rec in MUSIC_RECOMMENDATIONS.items():
        emotion_name = EMOTION_MAP.get(label_key, {}).get("name", label_key)
        emoji        = EMOTION_MAP.get(label_key, {}).get("emoji", "🎵")
        with st.expander(f"{emoji} {emotion_name}"):
            for i, t in enumerate(rec["tracks"]):
                st.markdown(f"**{i+1}.** {t['title']}  \n`{t['artist']}` · {t['genre']}")

    st.divider()
    st.markdown("### Model Info")
    st.markdown("**Weights file:** `model.emotion`")
    st.markdown("**Architecture:** Wav2Vec2ForSequenceClassification")
    st.markdown("**Hidden size:** 768  |  **Heads:** 12")
    st.markdown("**Labels:** 5  |  **Sample rate:** 16 000 Hz")
    if MODEL_WEIGHTS.exists():
        size_mb = MODEL_WEIGHTS.stat().st_size / 1e6
        st.markdown(f"**File size:** {size_mb:.1f} MB")
    else:
        st.warning("model.emotion not found in current directory"),