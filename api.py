from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import librosa

# ML modules
from emotion import detect_emotion
from feature_extraction import extract_features
from speech import speech_to_text
from keywords import detect_keywords

app = Flask(__name__)

# Load model
model, feature_length = pickle.load(open("model.pkl", "rb"))


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

        # ---------------------------
        # 🔥 FAST AUDIO LOAD (ALL FORMATS)
        # ---------------------------
        try:
            data, sr = librosa.load(filepath, sr=16000)
        except:
            return jsonify({"error": "Audio format not supported"})

        if len(data) == 0:
            return jsonify({"error": "Empty audio"})

        # ---------------------------
        # 🎧 BASIC AUDIO FEATURES (FAST)
        # ---------------------------
        mean = float(np.mean(data))
        std = float(np.std(data))
        audio_features = [mean, std]

        # ---------------------------
        # 🧠 SPEECH TO TEXT (SAFE)
        # ---------------------------
        try:
            text = speech_to_text(filepath)
        except:
            text = ""

        # ---------------------------
        # 🔑 KEYWORDS (SAFE)
        # ---------------------------
        try:
            keyword_score, words = detect_keywords(text) if text else (0, [])
        except:
            keyword_score, words = 0, []

        # ---------------------------
        # 😊 EMOTION (SAFE)
        # ---------------------------
        try:
            emotion, emotion_score = detect_emotion(text) if text else ("neutral", 0)
            emotion_score = int(emotion_score)
        except:
            emotion, emotion_score = "neutral", 0

        # ---------------------------
        # 🎯 COMBINE FEATURES
        # ---------------------------
        features = audio_features + [keyword_score, emotion_score]

        # Fix length for model
        if len(features) < feature_length:
            features += [0] * (feature_length - len(features))
        elif len(features) > feature_length:
            features = features[:feature_length]

        # ---------------------------
        # 🤖 MODEL PREDICTION (SAFE)
        # ---------------------------
        try:
            prediction = model.predict([features])[0]
        except:
            prediction = "UNKNOWN"

        # ---------------------------
        # ✅ FINAL RESPONSE
        # ---------------------------
        return jsonify({
            "prediction": str(prediction),
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