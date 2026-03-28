import streamlit as st
import pickle
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import time
import streamlit.components.v1 as components
import pandas as pd
import os
from speech import speech_to_text
from keywords import detect_keywords
from emotion import detect_emotion
from feature_extraction import extract_features
import textblob
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SOC Dashboard", layout="wide")

# ---------------- FEATHER ICONS ----------------
st.markdown("""
<script src="https://unpkg.com/feather-icons"></script>
""", unsafe_allow_html=True)

# ---------------- CYBER UI ----------------
st.markdown("""
<style>
/* FULL BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0a0f1c, #000000);
    color: #00FFAA;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #00FFAA;
}

/* TEXT */
h1, h2, h3 {
    color: #00FFAA !important;
}

/* GLOW EFFECT */
.glow {
    text-shadow: 0 0 10px #00FFAA, 0 0 20px #00FFAA;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("""
<h1 class="glow" style='text-align:center; font-size:70px;'>
AI CYBERSECURITY SOC
</h1>
<p style='text-align:center; color:#888; font-size:18px;'>
Real-Time Threat Intelligence System
</p>
<hr style='border:1px solid #00FFAA'>
""", unsafe_allow_html=True)

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = []

    def recv(self, frame: av.AudioFrame):
        self.audio_data.append(frame.to_ndarray())
        print("Receiving audio frame")
        return frame

# ---------------- FUNCTIONS ----------------
def show_metric(title, value, icon_name ,color):
    st.markdown(f"""
    <div style="
        background:#020617;
        padding:25px;
        border-radius:12px;
        border:1px solid {color};
        box-shadow: 0 0 20px {color};
        text-align:center;
        font-family:monospace;
    ">
        <h4 style="color:#888;">{title}</h4>
        <h1 style="color:{color};">{value}</h1>
    </div>
    """, unsafe_allow_html=True)

def plot_waveform(audio_path):
    sr, y = wavfile.read(audio_path)
    fig, ax = plt.subplots()
    ax.plot(y)
    st.pyplot(fig)

#def plot_spectrogram(audio_path):
 #   y, sr = librosa.load(audio_path)
 #   S = librosa.stft(y)
 #   S_db = librosa.amplitude_to_db(abs(S))
 #   fig, ax = plt.subplots()
 #   img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
 #   fig.colorbar(img, ax=ax)
 #   st.pyplot(fig)

def calculate_threat(keyword_score, emotion_score, prediction):
    score = keyword_score + emotion_score
    if prediction == "SCAM_CALLS":
        score += 50
    if keyword_score > 20:
        score += 20
    if prediction == "SCAM_CALLS" and score < 50:
        score = 50

    if score >= 80:
        return score, "HIGH"
    elif score >= 40:
        return score, "MEDIUM"
    else:
        return score, "LOW"

def record_audio(filename="recorded.wav", duration=5):

    webrtc_ctx = webrtc_streamer(
       key="record",
       audio_processor_factory=AudioProcessor,
       rtc_configuration=RTC_CONFIGURATION,
       media_stream_constraints={"audio": True, "video": False},
    )

    if st.button("Stop Recording"):
        if webrtc_ctx.audio_processor:
            audio_chunks = webrtc_ctx.audio_processor.audio_data

            if len(audio_chunks) > 0:
                audio_np = np.concatenate(audio_chunks, axis=0)
                audio_np = audio_np.flatten()
                audio_np = (audio_np * 32767).astype(np.int16)

                from scipy.io.wavfile import write
                write(filename, 16000, audio_np)

                return filename

    return None

def record_chunk(filename="temp.wav", duration=2):
    webrtc_ctx = webrtc_streamer(
       key="live",
       audio_processor_factory=AudioProcessor,
       rtc_configuration=RTC_CONFIGURATION,
       media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx.audio_processor:
        audio_chunks = webrtc_ctx.audio_processor.audio_data

        if len(audio_chunks) > 5:
            audio_np = np.concatenate(audio_chunks, axis=0)
            audio_np = audio_np.flatten()
            audio_np = (audio_np * 32767).astype(np.int16)

            from scipy.io.wavfile import write
            write(filename, 16000, audio_np)
            
            webrtc_ctx.audio_processor.audio_data = []

            return filename
    

    return None

# ---------------- LOAD MODEL ----------------
model, expected_features = pickle.load(open("model.pkl", "rb"))

# -------- HISTORY FILE --------
HISTORY_FILE = "history.csv"

if not os.path.exists(HISTORY_FILE):
    df = pd.DataFrame(columns=["Time", "Text", "Prediction", "Score", "Level", "Confidence"])
    df.to_csv(HISTORY_FILE, index=False)

# ---------------- SIDEBAR ----------------
st.sidebar.title("SOC Controls")

mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Audio", "Record Audio", "Live Detection"]
)

if "running" not in st.session_state:
    st.session_state.running = False


# ---------------- MODE ----------------
if mode == "Upload Audio":
    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
    if uploaded_file:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.audio_path = "temp.wav"

elif mode == "Record Audio":
    with st.expander("ℹ️ Instructions"):    
        st.markdown("""
        ### 🎤 How to Record
        1. Click **START** in the mic box  
        2. Speak clearly for a few seconds  
        3. Click **Stop Recording**  
        4. Then click **STOP** in the mic box  
        """)     

    audio_file = record_audio()
    
    if audio_file:
        st.session_state.audio_path = audio_file
        st.audio(audio_file)    

elif mode == "Live Detection":
    with st.expander("ℹ️ Instructions"):    
        st.markdown("""
        ### 🔴 Live Detection Instructions
        1. Click **Start Live**  
        2. Click **START** in the mic box  
        3. Speak continuously  
        4. System will auto-analyze every few seconds  
        5. Click **Stop Live** when done  
        """)    

    col1, col2 = st.columns(2)
    if col1.button("Start Live"):
        st.session_state.running = True
    if col2.button("Stop Live"):
        st.session_state.running = False

# ---------------- LIVE ----------------
if st.session_state.running:
    st.warning("LIVE DETECTION ACTIVE")
    audio_file = record_chunk()

    if audio_file:
        st.session_state.audio_path = audio_file
        st.audio(audio_file)

st.markdown("""
<div style="
    color:#00FFAA;
    font-family:monospace;
    animation: blink 1s infinite;
">
Scanning incoming audio stream...
</div>

<style>
@keyframes blink {
    50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ---------------- PROCESS ----------------
if st.session_state.audio_path is not None:
    audio_path = st.session_state.audio_path

    st.markdown("### <i data-feather='activity'></i> Audio Analysis", unsafe_allow_html=True)
    st.markdown("<script>feather.replace()</script>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("###  Waveform Analysis")
        plot_waveform(audio_path)
    with col2:
        st.markdown("### Spectrogram Analysis")
        plot_spectrogram(audio_path)

    st.markdown("### <i data-feather='cpu'></i> AI Processing", unsafe_allow_html=True)
    st.markdown("<script>feather.replace()</script>", unsafe_allow_html=True)
    
    audio_features = extract_features(audio_path)
    text = speech_to_text(audio_path)

    st.markdown(f"<i data-feather='file-text'></i> {text}", unsafe_allow_html=True)

    keyword_score, words = detect_keywords(text)
    st.markdown(f"<i data-feather='key'></i> {words}", unsafe_allow_html=True)

    emotion, emotion_score = detect_emotion(text)
    st.markdown(f"<i data-feather='smile'></i> {emotion}", unsafe_allow_html=True)

    st.markdown("<script>feather.replace()</script>", unsafe_allow_html=True)

    combined_features = list(audio_features) + [keyword_score, emotion_score]

    if len(combined_features) != expected_features:
        st.error("Feature mismatch!")
        st.stop()

    features = np.array(combined_features).reshape(1, -1)
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = round(max(proba) * 100, 2)

    threat_score, level = calculate_threat(keyword_score, emotion_score, prediction)

    from datetime import datetime

    new_data = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Text": text,
        "Prediction": prediction,
        "Score": threat_score,
        "Level": level,
        "Confidence": confidence
    }

    df = pd.DataFrame([new_data])
    df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)    
    
    st.session_state.audio_path = None

    st.markdown("### 🖥️ Threat Logs")

    st.code(f"""
    > Audio Captured
    > Speech Converted
    > Keywords Detected: {words}
    > Emotion: {emotion}
    > Model Prediction: {prediction}
    > Threat Score: {threat_score}
    > Risk Level: {level}
    """, language="bash")

    # COLORS
    if level == "HIGH":
        color = "red"
    elif level == "MEDIUM":
        color = "orange"
    else:
        color = "#00FFAA"

    st.markdown("### <i data-feather='alert-triangle'></i> Threat Dashboard", unsafe_allow_html=True)
    st.markdown("<script>feather.replace()</script>", unsafe_allow_html=True)

    col1, col2, col3 ,col4= st.columns(4)

    with col1:
        show_metric("Prediction", prediction, "cpu", color)
    with col2:
        show_metric("Threat Score", threat_score, "activity", color)
    with col3:
        show_metric("Risk Level", level, "alert-triangle", color)
    with col4:   
        show_metric("Confidence", f"{confidence}%", "shield", color)

    # ALERT BOX
    if level == "HIGH":
        st.error("CRITICAL SCAM DETECTED")
    elif level == "MEDIUM":
        st.warning("SUSPICIOUS CALL")
    else:
        st.success("SAFE CALL")

    if st.session_state.running:
        time.sleep(1)
        st.rerun()

    # -------- SOC LOGS --------
st.markdown("## 🗂️ Threat Logs")

df = pd.read_csv(HISTORY_FILE)

st.dataframe(df.tail(10), width="stretch")


# -------- THREAT GRAPH --------
st.markdown("##  Threat Trend")

st.line_chart(df["Score"])

st.download_button(
    "⬇️ Download Full Report",
    data=df.to_csv(index=False),
    file_name="scam_detection_report.csv",
    mime="text/csv"
)