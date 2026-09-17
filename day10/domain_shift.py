import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


SEED = 42
rng = np.random.default_rng(SEED)

N_SAMPLES = 6000
SEQ_LEN = 100


# ============================================================
# 1. Motion generator
#
# class 0: slow walking-like periodic motion
# class 1: fast running-like periodic motion
#
# 중요한 점:
# "움직임 자체"는 Phone A/B에서 동일하다.
# 달라지는 건 sensor 특성뿐.
# ============================================================

def generate_motion(label, n_samples):

    t = np.linspace(
        0,
        1,
        SEQ_LEN
    )

    samples = []

    for _ in range(n_samples):

        phase = rng.uniform(
            0,
            2 * np.pi
        )

        amplitude_jitter = rng.normal(
            1.0,
            0.08
        )

        if label == 0:

            # walking
            freq = rng.normal(
                2.0,
                0.15
            )

            amplitude = (
                1.0
                * amplitude_jitter
            )

        else:

            # running
            freq = rng.normal(
                4.0,
                0.20
            )

            amplitude = (
                1.5
                * amplitude_jitter
            )


        signal = (
            amplitude
            * np.sin(
                2 * np.pi * freq * t
                + phase
            )
        )


        # 실제 사람 움직임 자체의 variation
        signal += rng.normal(
            0,
            0.05,
            SEQ_LEN
        )

        samples.append(signal)


    return np.array(samples)


# ============================================================
# 2. Sensor model
#
# measured =
#
#     scale * true_signal
#     + bias
#     + sensor noise
#
# ============================================================

def apply_phone_sensor(
    signals,
    scale,
    bias,
    noise_std
):

    measured = (
        scale * signals
        + bias
        + rng.normal(
            0,
            noise_std,
            signals.shape
        )
    )

    return measured


# ============================================================
# 3. Generate ground-truth motion
# ============================================================

half = N_SAMPLES // 2

walk = generate_motion(
    0,
    half
)

run = generate_motion(
    1,
    half
)


X_true = np.concatenate([
    walk,
    run
])

y = np.concatenate([
    np.zeros(half),
    np.ones(half)
]).astype(int)


# shuffle
indices = rng.permutation(
    N_SAMPLES
)

X_true = X_true[indices]
y = y[indices]


# ============================================================
# 4. Phone A / Phone B
#
# 같은 X_true인데
# sensor 특성만 다르다.
# ============================================================

X_A = apply_phone_sensor(
    X_true,
    scale=1.00,
    bias=0.20,
    noise_std=0.08
)


X_B = apply_phone_sensor(
    X_true,
    scale=1.35,
    bias=-0.45,
    noise_std=0.15
)


# ============================================================
# 5. Train / test split
# ============================================================

split = int(
    0.7 * N_SAMPLES
)

train_idx = np.arange(
    0,
    split
)

test_idx = np.arange(
    split,
    N_SAMPLES
)


X_A_train = X_A[train_idx]
X_A_test = X_A[test_idx]

X_B_test = X_B[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


# ============================================================
# 6. Feature extractor
#
# RAW statistics:
#
# mean
# std
# max
# min
# RMS
#
# 여기에는 device bias / scale 정보도 강하게 섞인다.
# ============================================================

def raw_features(X):

    mean = X.mean(
        axis=1
    )

    std = X.std(
        axis=1
    )

    maximum = X.max(
        axis=1
    )

    minimum = X.min(
        axis=1
    )

    rms = np.sqrt(
        np.mean(
            X ** 2,
            axis=1
        )
    )

    return np.column_stack([
        mean,
        std,
        maximum,
        minimum,
        rms
    ])


# ============================================================
# 7. Domain-invariant-ish feature
#
# sample별로
#
# x' = (x - mean) / std
#
# 하면
#
# constant bias
# multiplicative scale
#
# 영향이 상당 부분 사라진다.
#
# 이후 frequency 관련 feature 추출.
# ============================================================

def invariant_features(X):

    mean = X.mean(
        axis=1,
        keepdims=True
    )

    std = X.std(
        axis=1,
        keepdims=True
    )

    X_norm = (
        X - mean
    ) / (
        std + 1e-8
    )


    # FFT
    spectrum = np.abs(
        np.fft.rfft(
            X_norm,
            axis=1
        )
    )


    # DC component 제외
    spectrum[:, 0] = 0


    dominant_bin = np.argmax(
        spectrum,
        axis=1
    )


    dominant_power = np.max(
        spectrum,
        axis=1
    )


    # shape-related statistics
    abs_mean = np.mean(
        np.abs(X_norm),
        axis=1
    )

    zero_crossings = np.sum(
        X_norm[:, 1:]
        * X_norm[:, :-1]
        < 0,
        axis=1
    )


    return np.column_stack([
        dominant_bin,
        dominant_power,
        abs_mean,
        zero_crossings
    ])


# ============================================================
# 8. Experiment helper
# ============================================================

def evaluate(
    feature_fn,
    name
):

    train_features = feature_fn(
        X_A_train
    )

    test_A_features = feature_fn(
        X_A_test
    )

    test_B_features = feature_fn(
        X_B_test
    )


    scaler = StandardScaler()

    train_features = (
        scaler.fit_transform(
            train_features
        )
    )

    test_A_features = (
        scaler.transform(
            test_A_features
        )
    )

    test_B_features = (
        scaler.transform(
            test_B_features
        )
    )


    model = LogisticRegression(
        max_iter=2000
    )

    model.fit(
        train_features,
        y_train
    )


    pred_A = model.predict(
        test_A_features
    )

    pred_B = model.predict(
        test_B_features
    )


    acc_A = accuracy_score(
        y_test,
        pred_A
    )

    acc_B = accuracy_score(
        y_test,
        pred_B
    )


    print(
        f"{name}"
    )

    print(
        f"  Phone A → Phone A : "
        f"{acc_A * 100:.2f}%"
    )

    print(
        f"  Phone A → Phone B : "
        f"{acc_B * 100:.2f}%"
    )

    print(
        f"  Domain gap        : "
        f"{(acc_A - acc_B) * 100:.2f}%p"
    )

    print()


# ============================================================
# 9. Run
# ============================================================

print(
    "=== DEVICE DOMAIN SHIFT ==="
)

print()

print(
    "Phone A:"
)

print(
    "  scale = 1.00"
)

print(
    "  bias  = +0.20"
)

print(
    "  noise = 0.08"
)

print()

print(
    "Phone B:"
)

print(
    "  scale = 1.35"
)

print(
    "  bias  = -0.45"
)

print(
    "  noise = 0.15"
)

print()


evaluate(
    raw_features,
    "RAW SENSOR FEATURES"
)

evaluate(
    invariant_features,
    "NORMALIZED / MOTION FEATURES"
)