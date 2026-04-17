# 🎙️ Echo Therapy – Emotion-Based Music Therapy System

**Real‑time emotion detection from voice | 89% accuracy on English & Urdu speech | Mood‑based music recommendations**

Echo Therapy is a web‑based intelligent system that listens to your voice, detects your emotional state (Happy, Sad, Angry, Neutral), and personalized music recommendations. Built with a fine‑tuned multilingual Wav2Vec2 model, it supports both English and Urdu speech, making it culturally adaptable.

---

## ✨ Features

- **Speech Emotion Recognition** – Uses a fine‑tuned Wav2Vec2 transformer model to classify emotions from voice input with **89% accuracy**.
- **Bilingual Support** – Works with both English and Urdu speech.
- **AI Emotional Companion** – Provides empathetic, context‑aware responses using Claude or GPT.
- **Mood‑Based Music Recommendations** – Suggests curated YouTube playlists and tracks that match your detected emotion.
- **Interactive Web Interface** – Built with Streamlit: upload audio, view emotion confidence charts, waveform visualization, and embedded music players.
- **Real‑time Performance** – Optimized for low‑latency inference on CPU or GPU.

---

## 🧠 Model Details

- **Architecture:** Wav2Vec2ForSequenceClassification (multilingual)
- **Fine‑tuned on:** Balanced dataset of English & Urdu emotional speech
- **Emotion classes:** Neutral, Happy, Sad, Angry, Fearful
- **Accuracy:** 89%
- **Input:** 16 kHz mono audio (WAV, MP3, FLAC, OGG, M4A)

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.8 or higher
- Git LFS (to download the model file) – [Install Git LFS](https://git-lfs.com)

### Step 1: Clone the repository
git clone https://github.com/saadf3819/EchoTherapy.git
cd EchoTherapy

### Step 2: Download the model file (Git LFS)
The model file model.emotion (378 MB) is stored with Git LFS. After cloning, run:
git lfs pull
This will download the actual model file (not just the pointer).

### Step 3: Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

### Step 4: Install dependencies
pip install -r requirements_assistant.txt
If you encounter issues, install manually:
pip install streamlit torch torchaudio librosa transformers plotly anthropic python-dotenv safetensors

### Step 6: Run the application
streamlit run llmapp.py
Open your browser at http://localhost:8501

## 🎮 How to Use
Upload an audio file – Supported formats: WAV, MP3, FLAC, OGG, M4A.

Click "Detect Emotion" – The model analyses the voice and displays:

Detected emotion with confidence score

Probability bar chart for all emotions

Audio waveform visualisation

Music Recommendations – Curated YouTube playlists tailored to your mood. Click "Play" to listen directly in the app.

📄 License
This project is for educational and research purposes.



