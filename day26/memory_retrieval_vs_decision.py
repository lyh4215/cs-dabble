memories = [
    {
        "activity": "botanical garden",
        "day": "weekday",
        "crowd": "quiet",
        "emotion": "calm",
        "valence": +0.9,
    },

    {
        "activity": "botanical garden",
        "day": "weekend",
        "crowd": "crowded",
        "emotion": "stressed",
        "valence": -0.8,
    },

    {
        "activity": "cat cafe",
        "day": "weekday",
        "crowd": "quiet",
        "emotion": "peaceful",
        "valence": +0.8,
    },

    {
        "activity": "music festival",
        "day": "weekend",
        "crowd": "crowded",
        "emotion": "exhausted",
        "valence": -0.9,
    },
]


# ================================================
# 1. Retrieval
#
# 여기서는 오직
# "현재 상황과 얼마나 관련 있는 기억인가?"
# 만 계산한다.
# ================================================

def context_match(
    memory,
    context,
):

    score = 0

    if memory["day"] == context["day"]:
        score += 1

    if memory["crowd"] == context["crowd"]:
        score += 1

    return score


def retrieve(
    context,
    min_match=2,
):

    results = []

    for m in memories:

        match = context_match(
            m,
            context,
        )

        if match >= min_match:

            results.append(
                m
            )

    return results


# ================================================
# 2. Decision
#
# retrieval된 기억의 결과가
# 실제로 좋았는지를 본다.
# ================================================

def decide(
    retrieved,
):

    positive = [
        m
        for m in retrieved
        if m["valence"] > 0
    ]


    if not positive:

        return None


    return max(
        positive,
        key=lambda m: m["valence"],
    )


# ================================================
# Test
# ================================================

queries = [
    {
        "name": "A",
        "context": {
            "day": "weekday",
            "crowd": "quiet",
        },
    },

    {
        "name": "B",
        "context": {
            "day": "weekend",
            "crowd": "crowded",
        },
    },
]


for q in queries:

    print(
        f"\n=== QUERY {q['name']} ==="
    )

    retrieved = retrieve(
        q["context"]
    )


    print(
        "Retrieved memories:"
    )

    for m in retrieved:

        print(
            f"  {m['activity']:18s}",
            f"{m['emotion']:10s}",
            f"{m['valence']:+.1f}",
        )


    decision = decide(
        retrieved
    )


    print(
        "\nDecision:"
    )

    if decision is None:

        print(
            "  No positive experience "
            "known for this context."
        )

        print(
            "  → abstain"
        )

    else:

        print(
            "  recommend:",
            decision["activity"],
        )