from flask import Flask, request, jsonify
import numpy as np
import wave
import pickle
import os

app = Flask(__name__)

# ✅ Load model
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

        # ✅ Read WAV properly
        with wave.open(filepath, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)

        # ✅ Basic feature (MATCH TRAINING STYLE LATER)
        feature = np.mean(data)

        # ⚠️ IMPORTANT: Create feature vector SAME SIZE as training
        features = [feature] * feature_length

        prediction = model.predict([features])[0]

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)