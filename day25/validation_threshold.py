import numpy as np

np.random.seed(0)


# ==================================================
# 1. synthetic calibrated relevance probabilities
# ==================================================

def make_dataset(n):

    # 실제 relevance 확률
    true_prob = np.random.beta(
        2,
        4,
        n,
    )

    labels = np.random.binomial(
        1,
        true_prob,
    )

    # calibrated model이라고 가정:
    # true probability 주변에서 약간 noisy한 예측
    scores = (
        true_prob
        + np.random.normal(
            0,
            0.08,
            n,
        )
    )

    scores = np.clip(
        scores,
        0,
        1,
    )

    return scores, labels


val_scores, val_labels = make_dataset(
    5000
)

test_scores, test_labels = make_dataset(
    5000
)


# ==================================================
# 2. metrics
# ==================================================

def evaluate(
    scores,
    labels,
    threshold,
):

    pred = (
        scores >= threshold
    )

    tp = np.sum(
        pred & (labels == 1)
    )

    fp = np.sum(
        pred & (labels == 0)
    )

    fn = np.sum(
        (~pred) & (labels == 1)
    )


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


    selected_ratio = (
        pred.mean()
    )


    return (
        precision,
        recall,
        selected_ratio,
    )


# ==================================================
# 3. choose threshold on validation set
#
# goal:
# recall >= 0.95
#
# among them:
# choose highest threshold
# ==================================================

TARGET_RECALL = 0.95

candidates = np.linspace(
    0,
    1,
    201,
)


valid = []

for threshold in candidates:

    precision, recall, selected = evaluate(
        val_scores,
        val_labels,
        threshold,
    )

    if recall >= TARGET_RECALL:

        valid.append(
            (
                threshold,
                precision,
                recall,
                selected,
            )
        )


best = max(
    valid,
    key=lambda x: x[0],
)


best_threshold = best[0]


print(
    "=== VALIDATION ==="
)

print(
    "chosen threshold:",
    best_threshold,
)

print(
    "precision:",
    best[1],
)

print(
    "recall:",
    best[2],
)

print(
    "selected ratio:",
    best[3],
)


# ==================================================
# 4. evaluate unseen test data
# ==================================================

p, r, selected = evaluate(
    test_scores,
    test_labels,
    best_threshold,
)


print(
    "\n=== TEST ==="
)

print(
    "precision:",
    p,
)

print(
    "recall:",
    r,
)

print(
    "selected ratio:",
    selected,
)


# ==================================================
# 5. compare with fixed threshold 0.5
# ==================================================

p05, r05, s05 = evaluate(
    test_scores,
    test_labels,
    0.5,
)


print(
    "\n=== FIXED 0.5 ==="
)

print(
    "precision:",
    p05,
)

print(
    "recall:",
    r05,
)

print(
    "selected ratio:",
    s05,
)


# ==================================================
# 6. compare several recall targets
# ==================================================

print(
    "\n=== DIFFERENT POLICIES ==="
)

for target in [
    0.90,
    0.95,
    0.99,
]:

    valid = []

    for threshold in candidates:

        p, r, s = evaluate(
            val_scores,
            val_labels,
            threshold,
        )

        if r >= target:

            valid.append(
                (
                    threshold,
                    p,
                    r,
                    s,
                )
            )


    best = max(
        valid,
        key=lambda x: x[0],
    )


    p_test, r_test, s_test = evaluate(
        test_scores,
        test_labels,
        best[0],
    )


    print(
        f"target_recall={target:.2f}",
        f"threshold={best[0]:.3f}",
        f"test_recall={r_test:.3f}",
        f"test_precision={p_test:.3f}",
        f"selected={s_test:.3f}",
    )