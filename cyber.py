def detect_keywords(text):
    scam_keywords = ["otp", "bank", "urgent", "account", "lottery", "kyc"]

    score = 0
    found_words = []

    for word in scam_keywords:
        if word in text:
            score += 10
            found_words.append(word)

    return score, found_words