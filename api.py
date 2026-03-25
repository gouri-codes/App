from flask import Flask, request, jsonify
import librosa
import numpy as np
import os

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=None)

        # Simple feature (example)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))

        # Dummy logic (replace with your model)
        if mfcc > 0:
            result = "Fraud Call"
        else:
            result = "Normal Call"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    app.run()