from flask import Flask, request, jsonify
import numpy as np
import pickle
import os


# ✅ Import ML modules
from emotion import detect_emotion
from feature_extraction import extract_features
from speech import speech_to_text
from keywords import detect_keywords

app = Flask(__name__)

# ✅ Load model ONCE
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
        filepath = "temp.wav"
        file.save(filepath)

        # 🎯 1. Extract audio features
        audio_features = extract_features(filepath)

        if audio_features is None:
            return jsonify({"error": "Feature extraction failed"})

        # 🎯 2. Speech to text
        text = speech_to_text(filepath)

        if text.strip() == "":
            return jsonify({"error": "No speech detected"})

        # 🎯 3. Keyword detection
        keyword_score, words = detect_keywords(text)

        # 🎯 4. Emotion detection
        emotion, emotion_score = detect_emotion(text)
        emotion_score = int(emotion_score)

        # 🎯 5. Combine features
        features = list(audio_features) + [keyword_score, emotion_score]

        # ⚠️ Ensure correct feature length
        if len(features) != feature_length:
            return jsonify({"error": "Feature length mismatch"})

        # 🎯 6. Prediction
        prediction = model.predict([features])[0]

        return jsonify({
            "prediction": prediction,
            "text": text,
            "emotion": emotion,
            "keywords": words
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)