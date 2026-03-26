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
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        filepath = "temp_audio"
        file.save(filepath)

        import numpy as np
        import soundfile as sf

        # 🔥 FAST LOAD (no librosa)
        try:
            data, sr = sf.read(filepath)
        except:
            return jsonify({"error": "Audio format not supported"})

        if len(data) == 0:
            return jsonify({"error": "Empty audio"})

        # 🎧 SIMPLE FEATURES (FAST)
        energy = float(np.mean(np.abs(data)))
        loudness = float(np.max(np.abs(data)))

        # 🧠 SAFE TEXT
        try:
            text = speech_to_text(filepath)
        except:
            text = ""

        # 🔑 SAFE KEYWORDS
        try:
            keyword_score, words = detect_keywords(text)
        except:
            keyword_score, words = 0, []

        # 😊 SAFE EMOTION
        try:
            emotion, emotion_score = detect_emotion(text)
            emotion_score = int(emotion_score)
        except:
            emotion, emotion_score = "neutral", 0

        # 🎯 SCORING
        score = 0

        if energy > 0.01:
            score += 20

        if loudness > 0.2:
            score += 20

        score += keyword_score
        score += emotion_score

        # 🚨 FINAL RESULT
        if score >= 60:
            prediction = "SCAM_CALLS"
        elif score >= 30:
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