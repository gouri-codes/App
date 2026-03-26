from flask import Flask, request, jsonify
import numpy as np
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
        filepath = "temp_audio"
        file.save(filepath)

        # ---------------------------
        # 🎧 SAFE AUDIO LOADING
        # ---------------------------
        try:
            import soundfile as sf
            data, sr = sf.read(filepath)

        except Exception as e:
            # 🔥 If recording file (3gp) → skip processing safely
            return jsonify({
                "prediction": "UNKNOWN",
                "text": "Recorded format not supported",
                "emotion": "neutral",
                "keywords": []
            })

        # ---------------------------
        # 🎯 SIMPLE FEATURE (FAST)
        # ---------------------------
        if len(data) == 0:
            return jsonify({"error": "Empty audio"})

        mean = float(np.mean(data))

        # ---------------------------
        # 🤖 SIMPLE PREDICTION (DEMO SAFE)
        # ---------------------------
        if abs(mean) > 0.01:
            prediction = "SCAM_CALLS"
        else:
            prediction = "NORMAL_CALL"

        # ---------------------------
        # ✅ FINAL RESPONSE
        # ---------------------------
        return jsonify({
            "prediction": prediction,
            "text": "Demo analysis complete",
            "emotion": "neutral",
            "keywords": ["sample"]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists("temp_audio"):
            os.remove("temp_audio")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)