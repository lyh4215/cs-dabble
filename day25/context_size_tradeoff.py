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
    "The company employs more than 500 engineers.",
    "Its products are used by electric vehicle manufacturers.",
    "Mina previously studied physics.",
    "The headquarters has a cafeteria and a gym.",
]


TARGET = 1


def score(query, text):

    vec = TfidfVectorizer(
        stop_words="english",
    )

    X = vec.fit_transform(
        [query, text]
    )

    return cosine_similarity(
        X[0],
        X[1],
    )[0, 0]


print(
    "=== CONTEXT SIZE ==="
)


for radius in range(
    0,
    7,
):

    start = max(
        0,
        TARGET - radius,
    )

    end = min(
        len(sentences),
        TARGET + radius + 1,
    )


    context = " ".join(
        sentences[
            start:end
        ]
    )


    s = score(
        query,
        context,
    )


    print(
        f"radius={radius}",
        f"sentences={end-start}",
        f"score={s:.3f}",
    )


    print(
        " ",
        context,
    )