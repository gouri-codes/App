from flask import Flask, request, jsonify
import librosa
import numpy as np
import os

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        filepath = "temp.wav"
        file.save(filepath)

        import librosa
        import numpy as np

        y, sr = librosa.load(filepath, sr=None)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))

        result = "Fraud Call" if mfcc > 0 else "Normal Call"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        import os
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

if __name__ == "__main__":
    app.run()