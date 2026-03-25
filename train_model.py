import os
from emotion import detect_emotion
from feature_extraction import extract_features
from speech import speech_to_text
from keywords import detect_keywords
from audio_visualization import plot_waveform, plot_spectrogram
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# dataset path
dataset_path = "C:/Users/gouri/Downloads/spam_call_detection/dataset-new"

features = []
labels = []
graph_count = 0

# ----------- DATA PROCESSING -----------
for label in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, label)

    if os.path.isdir(folder_path):

        for file in os.listdir(folder_path):

            file_path = os.path.join(folder_path, file)

            if file.endswith(".wav"):

                print("Processing:", file_path)
               
                if graph_count < 3:
                    plot_waveform(file_path)
                    plot_spectrogram(file_path)
                    graph_count += 1

                # audio features
                audio_features = extract_features(file_path)

                if audio_features is None:
                    continue

                # speech to text
                text = speech_to_text(file_path)

                print("Detected Text:", text)

                if text.strip() == "":
                    continue

                # keyword detection
                keyword_score, words = detect_keywords(text)
                emotion, emotion_score = detect_emotion(text)
                emotion_score = int(emotion_score)
                print(f"Detected Emotion: {emotion} (Score: {emotion_score})")
                print("Keywords Found:", words)

                # combine features
                combined_features = list(audio_features) + [keyword_score, emotion_score]

                features.append(combined_features)
                labels.append(label)


# ----------- CHECK DATA -----------
print("\nTotal samples:", len(features))

if len(features) == 0:
    print("No valid data found. Check dataset.")
    exit()


# ----------- TRAIN MODEL -----------
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2
)

model = RandomForestClassifier()
model.fit(X_train, y_train)


# ----------- EVALUATION -----------
predictions = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, predictions))


# ----------- THREAT FUNCTION -----------
def calculate_threat(keyword_score, emotion_score, prediction):

    keyword_score = int(keyword_score)
    emotion_score = int(emotion_score)

    score = keyword_score + emotion_score

    if prediction == "SCAM_CALLS":
        score += 50

    if keyword_score > 20:
        score += 20

    if emotion_score > 20:
        score += 20

    if score < 0:
        score = 0

    if prediction == "SCAM_CALLS" and score < 50:
        score = 50

    if score >= 100:
        level = "CRITICAL"
    elif score >= 80:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level
# ----------- FINAL REPORT -----------
for i in range(len(X_test)):

    prediction = model.predict([X_test[i]])[0]

    keyword_score = X_test[i][-2]
    emotion_score = X_test[i][-1]

    score, level = calculate_threat(keyword_score, emotion_score, prediction)

    print("\n--- Call Analysis Report ---")
    print("Prediction:", prediction)
    print("Keyword Score:", keyword_score)
    print("Emotion Score:", emotion_score)
    print("Threat Score:", score)
    print("Risk Level:", level)
    

import pickle

# Save model + feature length
pickle.dump((model, len(combined_features)), open("model.pkl", "wb"))

print("✅ Model + feature size saved")

