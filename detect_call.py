from feature_extraction import extract_features
import pickle

model = pickle.load(open("model.pkl","rb"))

file = "test_call.mp3"

features = extract_features(file)

result = model.predict([features])

print("Call Type:", result)