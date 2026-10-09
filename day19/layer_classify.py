import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


rng = np.random.default_rng(0)


# --------------------------------------------------
# 각 layer가 만드는 "가상의 EM 특성"
# mean amplitude, noise std, duration range
# --------------------------------------------------

LAYER_PROFILES = {
    "conv": {
        "mean": 1.8,
        "std": 0.25,
        "duration": (90, 140),
    },

    "relu": {
        "mean": 0.7,
        "std": 0.12,
        "duration": (30, 70),
    },

    "pool": {
        "mean": 1.0,
        "std": 0.18,
        "duration": (40, 80),
    },

    "fc": {
        "mean": 1.5,
        "std": 0.20,
        "duration": (60, 100),
    },
}


def generate_segment(layer):
    profile = LAYER_PROFILES[layer]

    length = rng.integers(
        profile["duration"][0],
        profile["duration"][1] + 1,
    )

    # 실행마다 amplitude가 약간 흔들린다고 가정
    amplitude = rng.normal(
        profile["mean"],
        0.12,
    )

    signal = rng.normal(
        amplitude,
        profile["std"],
        length,
    )

    return signal


def extract_features(signal):
    """
    공격자가 raw waveform 전체 대신
    segment의 통계적 특징을 사용한다고 가정.
    """

    mean = np.mean(signal)
    std = np.std(signal)
    duration = len(signal)

    # signal power
    power = np.mean(signal ** 2)

    # 변화량 크기
    diff_energy = np.mean(
        np.diff(signal) ** 2
    )

    return [
        mean,
        std,
        duration,
        power,
        diff_energy,
    ]


# --------------------------------------------------
# Dataset 생성
# --------------------------------------------------

X = []
y = []

SAMPLES_PER_LAYER = 500

for layer in LAYER_PROFILES:

    for _ in range(SAMPLES_PER_LAYER):

        signal = generate_segment(layer)

        features = extract_features(signal)

        X.append(features)
        y.append(layer)


X = np.array(X)
y = np.array(y)


# --------------------------------------------------
# Train / Test split
# --------------------------------------------------

indices = rng.permutation(len(X))

split = int(len(X) * 0.8)

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]


# --------------------------------------------------
# Classifier
# --------------------------------------------------

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=0,
)

clf.fit(
    X_train,
    y_train,
)


pred = clf.predict(X_test)


print("=== Classification report ===")

print(
    classification_report(
        y_test,
        pred,
        digits=3,
    )
)


print("=== Confusion matrix ===")

labels = [
    "conv",
    "relu",
    "pool",
    "fc",
]

print(
    confusion_matrix(
        y_test,
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