SMALLTALK_KEYWORDS = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "good morning", "good evening", "good night", "how are you",
    "what's up", "wassup", "sup", "ok", "okay", "cool", "great",
    "nice", "awesome", "welcome", "please", "sorry", "help"
}


def classify_query(query: str) -> str:
    q = query.lower().strip()

    for keyword in SMALLTALK_KEYWORDS:
        if keyword in q:
            return "smalltalk"

    words = q.split()
    if len(words) <= 2 and "?" not in q:
        return "smalltalk"

    return "knowledge"