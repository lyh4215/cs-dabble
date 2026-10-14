cases = [
    {
        "name": "simple",
        "scores": [
            0.91,   # gold
            0.54,   # distractor인데 꽤 높음
            0.31,
            0.18,
            0.12,
        ],
        "gold": {0},
    },

    {
        "name": "two-hop",
        "scores": [
            0.84,   # gold
            0.43,
            0.61,   # gold
            0.57,   # distractor
            0.20,
        ],
        "gold": {0, 2},
    },

    {
        "name": "complex",
        "scores": [
            0.88,   # gold
            0.69,   # gold
            0.52,   # distractor
            0.58,   # gold
            0.34,
            0.48,   # gold but classifier uncertain
        ],
        "gold": {0, 1, 3, 5},
    },
]


def adaptive(scores, threshold):

    return {
        i
        for i, score in enumerate(scores)
        if score >= threshold
    }


def evaluate(selected, gold):

    tp = len(selected & gold)
    fp = len(selected - gold)
    fn = len(gold - selected)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    return precision, recall


thresholds = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
]


print(
    "threshold | precision | recall | selected"
)

print(
    "-" * 45
)


for threshold in thresholds:

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_selected = 0

    for case in cases:

        selected = adaptive(
            case["scores"],
            threshold,
        )

        gold = case["gold"]

        total_tp += len(
            selected & gold
        )

        total_fp += len(
            selected - gold
        )

        total_fn += len(
            gold - selected
        )

        total_selected += len(
            selected
        )


    precision = (
        total_tp
        / (total_tp + total_fp)
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
    )

    avg_selected = (
        total_selected
        / len(cases)
    )


    print(
        f"{threshold:9.2f} | "
        f"{precision:9.3f} | "
        f"{recall:6.3f} | "
        f"{avg_selected:8.2f}"
    )


print(
    "\n=== PER QUERY @ threshold=0.5 ==="
)

threshold = 0.5

for case in cases:

    selected = adaptive(
        case["scores"],
        threshold,
    )

    precision, recall = evaluate(
        selected,
        case["gold"],
    )

    print(
        case["name"],
        "gold=",
        sorted(case["gold"]),
        "selected=",
        sorted(selected),
        f"precision={precision:.2f}",
        f"recall={recall:.2f}",
    )