from flask import Flask, request, jsonify
import numpy as np
import soundfile as sf
import os

app = Flask(__name__)

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

        # ✅ Use soundfile instead of librosa
        data, samplerate = sf.read(filepath)

        # Simple feature
        feature = np.mean(data)

        result = "Fraud Call" if feature > 0 else "Normal Call"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

if __name__ == "__main__":
    app.run()