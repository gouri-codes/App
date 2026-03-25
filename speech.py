import speech_recognition as sr

def speech_to_text(file_path):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return text.lower()

    except Exception as e:
        return ""