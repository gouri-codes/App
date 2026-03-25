from textblob import TextBlob

def detect_emotion(text):

    text = text.lower()

    fear_words = [
        "urgent", "immediately", "blocked", "suspended",
        "fraud", "risk", "hacked", "attack", "warning"
    ]

    anger_words = [
        "pay now", "last warning", "final notice",
        "legal action", "penalty"
    ]

    score = 0
    emotion = "NEUTRAL"

    for word in fear_words:
        if word in text:
            score += 15
            emotion = "FEAR"

    for word in anger_words:
        if word in text:
            score += 20
            emotion = "ANGER"

    if score == 0:
        emotion = "CALM"
        score = -5

    return emotion, int(score)