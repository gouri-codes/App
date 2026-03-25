def detect_keywords(text):
    scam_keywords = [
        "otp", "one time password", "bank", "account",
        "verify", "verification", "urgent", "immediately",
        "lottery", "prize", "winner", "kyc",
        "suspend", "blocked", "fraud", "security",
        "credit card", "debit card", "pin","hacked", "malware", "urgent", "payment", "card", "verify", "login", "security alert"
    ]

    score = 0
    found_words = []

    for word in scam_keywords:
        if word in text:
            score += 10
            found_words.append(word)

    return score, found_words