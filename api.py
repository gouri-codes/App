from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import wave

# ML modules
from emotion import detect_emotion
from feature_extraction import extract_features
from speech import speech_to_text
from keywords import detect_keywords

app = Flask(__name__)

model, feature_length = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "API is running"

# 🔥 FAST MODE (NO TIMEOUT)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        filepath = "temp.wav"
        file.save(filepath)

        with wave.open(filepath, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)

        mean = np.mean(data)
        std = np.std(data)

        features = [mean, std]

        if len(features) < feature_length:
            features += [0] * (feature_length - len(features))

        prediction = model.predict([features])[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")


# 🧠 FULL ANALYSIS (KEYWORDS + EMOTION)
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["file"]
        filepath = "temp.wav"
        file.save(filepath)

        # FULL PIPELINE
        audio_features = extract_features(filepath)
        text = speech_to_text(filepath)

        if text.strip() == "":
            return jsonify({"error": "No speech detected"})

        keyword_score, words = detect_keywords(text)
        emotion, emotion_score = detect_emotion(text)

        features = list(audio_features) + [keyword_score, int(emotion_score)]

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