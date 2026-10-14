from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


history = [
    (
        "Last weekend I went to a crowded music festival. "
        "I was exhausted afterward."
    ),

    (
        "The botanical garden made me feel calm and refreshed."
    ),

    (
        "At the cat cafe I felt peaceful and comfortable."
    ),

    (
        "I bought a new keyboard for work."
    ),
]


query = (
    "I'm exhausted after work. "
    "What should I do this weekend?"
)


# ==================================================
# 1. Raw text retrieval
# ==================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
)

X = vectorizer.fit_transform(
    [query] + history
)

scores = cosine_similarity(
    X[0],
    X[1:],
)[0]


ranking = scores.argsort()[::-1]


print(
    "=== RAW TEXT RETRIEVAL ==="
)

for i in ranking:

    print(
        f"score={scores[i]:.3f}",
        history[i],
    )


# ==================================================
# 2. Structured long-term memory
#
# 실제 시스템에서는 NLP/LLM 등이
# conversation에서 이런 구조를 추출한다고 생각.
# ==================================================

memory = [
    {
        "activity": "music festival",
        "emotion": "exhausted",
        "valence": -0.9,
    },

    {
        "activity": "botanical garden",
        "emotion": "calm",
        "valence": +0.9,
    },

    {
        "activity": "cat cafe",
        "emotion": "peaceful",
        "valence": +0.8,
    },
]


# ==================================================
# 3. Agent goal
#
# 현재 사용자는 exhausted 상태.
# → positive / calming experience를 찾는다.
# ==================================================

candidates = sorted(
    memory,
    key=lambda m: m["valence"],
    reverse=True,
)


print(
    "\n=== STRUCTURED MEMORY ==="
)

for m in candidates:

    print(
        f"{m['activity']:18s}",
        f"emotion={m['emotion']:10s}",
        f"valence={m['valence']:+.1f}",
    )


best = candidates[0]


print(
    "\n=== AGENT DECISION ==="
)

print(
    "recommend:",
    best["activity"],
)

print(
    "because past emotion was:",
    best["emotion"],
)