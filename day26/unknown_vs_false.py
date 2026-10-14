memories = {
    "botanical garden": {
        "known": True,
        "valence": +0.9,
    },

    "music festival": {
        "known": True,
        "valence": -0.9,
    },

    # museum은 아예 없음
}


def preference(activity):

    if activity not in memories:
        return "UNKNOWN"

    valence = memories[
        activity
    ]["valence"]

    if valence > 0:
        return "POSITIVE"

    elif valence < 0:
        return "NEGATIVE"

    else:
        return "NEUTRAL"


activities = [
    "botanical garden",
    "music festival",
    "museum",
]


print(
    "=== KNOWLEDGE STATE ==="
)

for activity in activities:

    print(
        f"{activity:18s}",
        preference(activity),
    )


# =========================================
# naive decision
#
# UNKNOWN을 NEGATIVE처럼 취급하는 실수
# =========================================

print(
    "\n=== NAIVE DECISION ==="
)

for activity in activities:

    state = preference(
        activity
    )

    recommend = (
        state == "POSITIVE"
    )

    print(
        f"{activity:18s}",
        "recommend"
        if recommend
        else "do not recommend",
    )


# =========================================
# knowledge-aware decision
# =========================================

print(
    "\n=== KNOWLEDGE-AWARE DECISION ==="
)

for activity in activities:

    state = preference(
        activity
    )

    if state == "POSITIVE":

        decision = "recommend"

    elif state == "NEGATIVE":

        decision = "avoid"

    elif state == "UNKNOWN":

        decision = (
            "unknown "
            "→ ask / explore"
        )

    else:

        decision = "neutral"


    print(
        f"{activity:18s}",
        decision,
    )