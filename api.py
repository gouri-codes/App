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
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        filepath = "temp_audio"
        file.save(filepath)

        import librosa
        import numpy as np

        # 🔥 FORCE SIMPLE LOAD ONLY
        data, sr = librosa.load(filepath, sr=16000)

        if len(data) == 0:
            return jsonify({"error": "Empty audio"})

        mean = float(np.mean(data))

        # 🔥 SIMPLE RULE (NO ML)
        if abs(mean) > 0.01:
            prediction = "SCAM_CALLS"
        else:
            prediction = "NORMAL_CALL"

        return jsonify({
            "prediction": prediction,
            "text": "demo text",
            "emotion": "neutral",
            "keywords": []
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp_audio"):
            os.remove("temp_audio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)