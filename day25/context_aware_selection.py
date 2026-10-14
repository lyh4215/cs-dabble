from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


query = (
    "What instrument does the person "
    "behind Acme Labs use?"
)

sentences = [
    "Acme Labs was founded by Mina Park in 2018.",
    "She performs jazz on the saxophone.",
    "The company develops battery software.",
    "Its office opened in 2020.",
]


# ==================================================
# similarity helper
# ==================================================

def similarity(query, texts):

    vectorizer = TfidfVectorizer(
        stop_words="english",
    )

    matrix = vectorizer.fit_transform(
        [query] + texts
    )

    q = matrix[0]

    docs = matrix[1:]

    return cosine_similarity(
        q,
        docs,
    )[0]


# ==================================================
# 1. Independent sentence selection
#
# 각 문장을 완전히 혼자 판단
# ==================================================

independent_score = similarity(
    query,
    sentences,
)


print(
    "=== INDEPENDENT ==="
)

for i, score in enumerate(
    independent_score
):

    print(
        f"S{i}:",
        f"{score:.3f}",
        sentences[i],
    )


# ==================================================
# 2. Context-aware selection toy
#
# 각 문장을 볼 때
# 이전 문장 + 현재 문장을 같이 본다.
#
# 실제 EXIT 구현 그 자체는 아니고
# context dependency의 효과만 재현
# ==================================================

contexts = []

for i in range(
    len(sentences)
):

    if i == 0:

        context = sentences[i]

    else:

        context = (
            sentences[i - 1]
            + " "
            + sentences[i]
        )

    contexts.append(
        context
    )


context_score = similarity(
    query,
    contexts,
)


print(
    "\n=== CONTEXT-AWARE ==="
)

for i, score in enumerate(
    context_score
):

    print(
        f"S{i}:",
        f"{score:.3f}",
        sentences[i],
    )


# ==================================================
# compare S1
# ==================================================

print(
    "\n=== TARGET SENTENCE S1 ==="
)

print(
    "independent:",
    independent_score[1],
)

print(
    "with context:",
    context_score[1],
)