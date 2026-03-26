from flask import Flask, request, jsonify
import numpy as np
import os
import librosa

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
        filepath = "temp_audio"

        file.save(filepath)

        # ✅ FIX: use librosa (supports .3gp)
        data, samplerate = librosa.load(filepath, sr=None)

        feature = np.mean(data)

        result = "Fraud Call" if feature > 0 else "Normal Call"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp_audio"):
            os.remove("temp_audio")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)