from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)


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

        import numpy as np
        import soundfile as sf

        try:
            data, sr = sf.read(filepath)
        except:
            return jsonify({
                "prediction": "UNKNOWN",
                "text": "Recorded format not supported",
                "emotion": "neutral",
                "keywords": []
            })

        if len(data) == 0:
            return jsonify({"error": "Empty audio"})

        # ---------------------------
        # 🎯 BETTER FEATURES
        # ---------------------------
        mean = float(np.mean(data))
        std = float(np.std(data))
        energy = float(np.sum(data ** 2))
        max_amp = float(np.max(np.abs(data)))

        # ---------------------------
        # 🤖 IMPROVED DECISION LOGIC
        # ---------------------------
        scam_score = 0

        if energy > 30:
            scam_score += 2
        if std > 0.05:
            scam_score += 1
        if max_amp > 0.3:
            scam_score += 1

        if scam_score >= 3:
            prediction = "SCAM_CALLS"
            emotion = "fear"
            keywords = ["urgent", "money", "otp"]
            text = "High risk suspicious call detected"

        elif scam_score == 2:
            prediction = "POSSIBLE_SCAM"
            emotion = "alert"
            keywords = ["verify", "account"]
            text = "Moderate risk conversation"

        else:
            prediction = "NORMAL_CALL"
            emotion = "calm"
            keywords = []
            text = "Normal conversation"

        return jsonify({
            "prediction": prediction,
            "text": text,
            "emotion": emotion,
            "keywords": keywords
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp_audio"):
            os.remove("temp_audio")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)