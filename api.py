from flask import Flask, request, jsonify
import numpy as np
import os
import librosa

app = Flask(__name__)

# ---------------------------
# 🔒 SAFE IMPORTS
# ---------------------------
try:
    from speech import speech_to_text
except:
    def speech_to_text(x):
        return ""

try:
    from keywords import detect_keywords
except:
    def detect_keywords(text):
        return 0, []

try:
    from emotion import detect_emotion
except:
    def detect_emotion(text):
        return "neutral", 0


@app.route("/")
def home():
    return "API is running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        filepath = "temp_audio"
        file.save(filepath)

        # 🎧 LOAD AUDIO
        data, sr = librosa.load(filepath, sr=16000)

        if len(data) == 0:
            return jsonify({"error": "Empty audio"})

        # ---------------------------
        # 🎧 AUDIO FEATURES
        # ---------------------------
        energy = float(np.mean(np.abs(data)))
        loudness = float(np.max(np.abs(data)))

        # ---------------------------
        # 🧠 SPEECH
        # ---------------------------
        text = speech_to_text(filepath)

        # ---------------------------
        # 🔑 KEYWORDS
        # ---------------------------
        keyword_score, words = detect_keywords(text)

        # ---------------------------
        # 😊 EMOTION
        # ---------------------------
        emotion, emotion_score = detect_emotion(text)
        emotion_score = int(emotion_score)

        # ---------------------------
        # 🎯 SMART SCORING
        # ---------------------------
        score = 0

        if energy > 0.02:
            score += 20

        if loudness > 0.3:
            score += 20

        score += keyword_score
        score += emotion_score

        # ---------------------------
        # 🚨 FINAL RESULT
        # ---------------------------
        if score >= 70:
            prediction = "SCAM_CALLS"
        elif score >= 40:
            prediction = "POSSIBLE_SCAM"
        else:
            prediction = "NORMAL_CALL"

        return jsonify({
            "prediction": prediction,
            "score": score,
            "text": text,
            "emotion": emotion,
            "keywords": words
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp_audio"):
            os.remove("temp_audio")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)