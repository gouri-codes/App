from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import librosa
import soundfile as sf

# ML modules
from emotion import detect_emotion
from feature_extraction import extract_features
from speech import speech_to_text
from keywords import detect_keywords

app = Flask(__name__)

# ✅ Load model once
model, feature_length = pickle.load(open("model.pkl", "rb"))


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

        # ---------------------------
        # 🎧 LOAD AUDIO (SAFE)
        # ---------------------------
        try:
            import soundfile as sf
            data, sr = sf.read(filepath)
        except:
            try:
                import librosa
                data, sr = librosa.load(filepath, sr=16000)
            except Exception as e:
                return jsonify({"error": "Audio read failed: " + str(e)})

        if len(data) == 0:
            return jsonify({"error": "Empty audio"})

        # ---------------------------
        # 🎯 BASIC FEATURES
        # ---------------------------
        mean = float(np.mean(data))
        std = float(np.std(data))
        audio_features = [mean, std]

        # ---------------------------
        # 🧠 SPEECH TO TEXT (SAFE)
        # ---------------------------
        try:
            text = speech_to_text(filepath)
        except Exception as e:
            text = ""
            print("Speech error:", e)

        # ---------------------------
        # 🔑 KEYWORDS (SAFE)
        # ---------------------------
        try:
            keyword_score, words = detect_keywords(text) if text else (0, [])
        except Exception as e:
            keyword_score, words = 0, []
            print("Keyword error:", e)

        # ---------------------------
        # 😊 EMOTION (SAFE)
        # ---------------------------
        try:
            emotion, emotion_score = detect_emotion(text) if text else ("neutral", 0)
            emotion_score = int(emotion_score)
        except Exception as e:
            emotion, emotion_score = "neutral", 0
            print("Emotion error:", e)

        # ---------------------------
        # 🤖 MODEL (SAFE)
        # ---------------------------
        try:
            features = audio_features + [keyword_score, emotion_score]

            if len(features) < feature_length:
                features += [0] * (feature_length - len(features))

            prediction = model.predict([features])[0]
        except Exception as e:
            print("Model error:", e)
            prediction = "UNKNOWN"

        return jsonify({
            "prediction": str(prediction),
            "text": text,
            "emotion": emotion,
            "keywords": words
        })

    except Exception as e:
        return jsonify({"error": "Server crashed: " + str(e)})

    finally:
        if os.path.exists("temp_audio"):
            os.remove("temp_audio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)