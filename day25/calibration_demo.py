import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ==================================================
# 1. underlying true relevance
# ==================================================

N = 10000

latent = np.random.normal(
    0,
    1,
    N,
)

true_prob = sigmoid(
    latent
)

labels = np.random.binomial(
    1,
    true_prob,
)


# ==================================================
# 2. pretend relevance classifier
#
# ranking은 거의 맞지만
# 너무 confident + positive 쪽으로 bias됨
# ==================================================

raw_logit = (
    3.0 * latent
    + 1.2
)

raw_prob = sigmoid(
    raw_logit
)


# ==================================================
# 3. split calibration / test
# ==================================================

idx = np.random.permutation(N)

cal_idx = idx[:5000]
test_idx = idx[5000:]


# ==================================================
# 4. Platt scaling
#
# raw logit -> calibrated probability
# ==================================================

calibrator = LogisticRegression()

calibrator.fit(
    raw_logit[
        cal_idx
    ].reshape(-1, 1),
    labels[
        cal_idx
    ],
)


calibrated_prob = (
    calibrator
    .predict_proba(
        raw_logit[
            test_idx
        ].reshape(-1, 1)
    )[:, 1]
)


raw_test = raw_prob[
    test_idx
]

y_test = labels[
    test_idx
]


# ==================================================
# 5. Brier score
#
# probability가 얼마나 맞는지
# 낮을수록 좋음
# ==================================================

raw_brier = brier_score_loss(
    y_test,
    raw_test,
)

cal_brier = brier_score_loss(
    y_test,
    calibrated_prob,
)


print(
    "=== BRIER SCORE ==="
)

print(
    "raw       :",
    raw_brier,
)

print(
    "calibrated:",
    cal_brier,
)


# ==================================================
# 6. threshold = 0.5
# ==================================================

raw_pred = (
    raw_test >= 0.5
)

cal_pred = (
    calibrated_prob >= 0.5
)


def metrics(pred, y):

    TP = np.sum(
        pred & (y == 1)
    )

    FP = np.sum(
        pred & (y == 0)
    )

    FN = np.sum(
        (~pred) & (y == 1)
    )

    precision = (
        TP
        / (TP + FP)
    )

    recall = (
        TP
        / (TP + FN)
    )

    return (
        precision,
        recall,
    )


raw_p, raw_r = metrics(
    raw_pred,
    y_test,
)

cal_p, cal_r = metrics(
    cal_pred,
    y_test,
)


print(
    "\n=== THRESHOLD 0.5 ==="
)

print(
    "raw:",
    f"precision={raw_p:.3f}",
    f"recall={raw_r:.3f}",
)

print(
    "calibrated:",
    f"precision={cal_p:.3f}",
    f"recall={cal_r:.3f}",
)


# ==================================================
# 7. reliability bins
#
# score가 0.8 근처인 sample들의
# 실제 positive 비율 확인
# ==================================================

def reliability(
    probs,
    y,
):

    bins = [
        (0.1, 0.3),
        (0.3, 0.5),
        (0.5, 0.7),
        (0.7, 0.9),
    ]

    for lo, hi in bins:

        mask = (
            (probs >= lo)
            & (probs < hi)
        )

        if mask.sum() == 0:
            continue

        predicted_mean = (
            probs[
                mask
            ].mean()
        )

        actual_rate = (
            y[
                mask
            ].mean()
        )

        print(
            f"{lo:.1f}-{hi:.1f}",
            f"pred={predicted_mean:.3f}",
            f"actual={actual_rate:.3f}",
            f"n={mask.sum()}",
        )


print(
    "\n=== RAW RELIABILITY ==="
)

reliability(
    raw_test,
    y_test,
)


print(
    "\n=== CALIBRATED RELIABILITY ==="
)

reliability(
    calibrated_prob,
    y_test,
)


print(
    "\n=== CALIBRATOR ==="
)

print(
    "coefficient:",
    calibrator.coef_[0][0],
)

print(
    "intercept:",
    calibrator.intercept_[0],
)