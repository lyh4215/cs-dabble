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


# ==================================================
# 1. naive aggregation
#
# activity별로 context를 전부 뭉개버림
# ==================================================

from collections import defaultdict


values = defaultdict(list)

for m in memories:

    values[
        m["activity"]
    ].append(
        m["valence"]
    )


print(
    "=== NAIVE MEMORY ==="
)

for activity, vals in values.items():

    avg = sum(vals) / len(vals)

    print(
        f"{activity:18s}",
        f"average={avg:+.2f}",
    )


# ==================================================
# 2. context-aware retrieval
# ==================================================

def context_score(
    memory,
    query_context,
):

    score = 0

    if (
        memory["day"]
        == query_context["day"]
    ):
        score += 1

    if (
        memory["crowd"]
        == query_context["crowd"]
    ):
        score += 1

    return score


def recommend(
    query_context,
):

    candidates = []

    for m in memories:

        context_match = context_score(
            m,
            query_context,
        )

        # context match를 우선하고,
        # 같은 context라면 positive experience 선호
        total_score = (
            2.0 * context_match
            + m["valence"]
        )

        candidates.append(
            (
                total_score,
                m,
            )
        )


    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )


    return candidates


# ==================================================
# Case A
# quiet weekday
# ==================================================

query_A = {
    "day": "weekday",
    "crowd": "quiet",
}


print(
    "\n=== QUERY A ==="
)

print(
    query_A
)

for score, m in recommend(
    query_A
):

    print(
        f"score={score:+.2f}",
        m["activity"],
        "/",
        m["emotion"],
    )


# ==================================================
# Case B
# crowded weekend
# ==================================================

query_B = {
    "day": "weekend",
    "crowd": "crowded",
}


print(
    "\n=== QUERY B ==="
)

print(
    query_B
)

for score, m in recommend(
    query_B
):

    print(
        f"score={score:+.2f}",
        m["activity"],
        "/",
        m["emotion"],
    )