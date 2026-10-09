import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


rng = np.random.default_rng(0)


PROFILES = {
    "1x1": {
        "mean": 1.20,
        "std": 0.16,
        "duration": (60, 90),
    },

    "3x3": {
        "mean": 1.55,
        "std": 0.22,
        "duration": (100, 150),
    },

    "5x5": {
        "mean": 1.85,
        "std": 0.28,
        "duration": (150, 220),
    },
}


def generate_trace(kernel):
    p = PROFILES[kernel]

    length = rng.integers(
        p["duration"][0],
        p["duration"][1] + 1,
    )

    # 실행마다 환경 변화
    scale = rng.normal(1.0, 0.08)

    mean = p["mean"] * scale

    signal = rng.normal(
        mean,
        p["std"],
        length,
    )

    return signal


def features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        len(signal),
        np.mean(signal ** 2),
        np.mean(np.diff(signal) ** 2),
    ]


X = []
y = []

for kernel in PROFILES:

    for _ in range(600):
        signal = generate_trace(kernel)

        X.append(
            features(signal)
        )

        y.append(kernel)


X = np.array(X)
y = np.array(y)


idx = rng.permutation(len(X))

split = int(len(X) * 0.8)

train = idx[:split]
test = idx[split:]


clf = RandomForestClassifier(
    n_estimators=200,
    random_state=0,
)

clf.fit(
    X[train],
    y[train],
)


pred = clf.predict(
    X[test]
)


print("=== Kernel-size inference ===")

print(
    classification_report(
        y[test],
        pred,
        digits=3,
    )
)


labels = [
    "1x1",
    "3x3",
    "5x5",
]

print("=== Confusion matrix ===")

print(
    confusion_matrix(
        y[test],
        pred,
        labels=labels,
    )
)


print("\n=== Feature importance ===")

names = [
    "mean",
    "std",
    "duration",
    "power",
    "diff_energy",
]

for name, importance in sorted(
    zip(
        names,
        clf.feature_importances_,
    ),
    key=lambda x: x[1],
    reverse=True,
):

    print(
        f"{name:12s}",
        f"{importance:.3f}",
    )