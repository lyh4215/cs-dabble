cases = [
    {
        "name": "simple",
        "scores": [
            0.95,
            0.41,
            0.22,
            0.16,
            0.09,
            0.05,
        ],
        "gold": {0},
    },

    {
        "name": "two-hop",
        "scores": [
            0.91,
            0.37,
            0.84,
            0.32,
            0.18,
            0.11,
        ],
        "gold": {0, 2},
    },

    {
        "name": "complex",
        "scores": [
            0.88,
            0.81,
            0.44,
            0.79,
            0.31,
            0.76,
        ],
        "gold": {0, 1, 3, 5},
    },
]


def fixed_top_k(scores, k):

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    return set(
        ranked[:k]
    )


def adaptive(scores, threshold=0.5):

    return {
        i
        for i, score in enumerate(scores)
        if score >= threshold
    }


def evaluate(
    selected,
    gold,
):

    evidence_recall = (
        len(selected & gold)
        / len(gold)
    )

    extra = len(
        selected - gold
    )

    return (
        evidence_recall,
        extra,
    )


methods = {
    "top-2": (
        lambda s:
        fixed_top_k(s, 2)
    ),

    "top-4": (
        lambda s:
        fixed_top_k(s, 4)
    ),

    "adaptive": (
        lambda s:
        adaptive(s, 0.5)
    ),
}


print(
    "=== PER QUERY ==="
)

for case in cases:

    print(
        f"\n[{case['name']}]"
    )

    print(
        "gold:",
        sorted(case["gold"]),
    )

    for name, fn in methods.items():

        selected = fn(
            case["scores"]
        )

        recall, extra = evaluate(
            selected,
            case["gold"],
        )

        print(
            f"{name:8s}",
            f"selected={sorted(selected)}",
            f"recall={recall:.2f}",
            f"extra={extra}",
        )


print(
    "\n=== AVERAGE ==="
)

for name, fn in methods.items():

    recalls = []
    extras = []
    counts = []

    for case in cases:

        selected = fn(
            case["scores"]
        )

        recall, extra = evaluate(
            selected,
            case["gold"],
        )

        recalls.append(
            recall
        )

        extras.append(
            extra
        )

        counts.append(
            len(selected)
        )

    print(
        f"{name:8s}",
        f"evidence_recall={sum(recalls)/len(recalls):.3f}",
        f"extra={sum(extras)/len(extras):.2f}",
        f"selected={sum(counts)/len(counts):.2f}",
    )