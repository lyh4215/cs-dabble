import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


SEED = 42
rng = np.random.default_rng(SEED)

N_SAMPLES = 6000
SEQ_LEN = 128


# ============================================================
# 1. Base 3-axis motion generator
#
# class 0 = walking-like
# class 1 = running-like
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

        if label == 0:
            freq = rng.normal(
                2.0,
                0.15
            )
            amp = rng.normal(
                1.0,
                0.08
            )

        else:
            freq = rng.normal(
                4.0,
                0.20
            )
            amp = rng.normal(
                1.5,
                0.10
            )


        # 실제 motion은 3축 모두에 일부 존재
        x = (
            amp
            * np.sin(
                2 * np.pi
                * freq
                * t
                + phase
            )
        )

        y = (
            0.35
            * amp
            * np.sin(
                2 * np.pi
                * freq
                * t
                + phase
                + 0.7
            )
        )

        z = (
            0.20
            * amp
            * np.sin(
                4 * np.pi
                * freq
                * t
                + phase
            )
        )


        sample = np.stack(
            [x, y, z],
            axis=1
        )


        # 사람 자체의 motion variation
        sample += rng.normal(
            0,
            0.04,
            sample.shape
        )

        samples.append(
            sample
        )


    return np.array(
        samples
    )


# ============================================================
# 2. Rotation matrix
# ============================================================

def rotation_matrix_xyz(
    rx,
    ry,
    rz
):

    cx, sx = (
        np.cos(rx),
        np.sin(rx)
    )

    cy, sy = (
        np.cos(ry),
        np.sin(ry)
    )

    cz, sz = (
        np.cos(rz),
        np.sin(rz)
    )


    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])


    return (
        Rz @ Ry @ Rx
    )


# ============================================================
# 3. Phone sensor model
#
# rotation
# + axis-specific scale
# + bias
# + noise
# ============================================================

def apply_phone_sensor(
    X,
    rotation,
    scale,
    bias,
    noise_std
):

    # orientation change
    rotated = (
        X @ rotation.T
    )


    # axis별 scale
    measured = (
        rotated
        * scale
    )


    # axis별 bias
    measured = (
        measured
        + bias
    )


    # noise
    measured += rng.normal(
        0,
        noise_std,
        measured.shape
    )


    return measured


# ============================================================
# 4. Generate ground truth
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


indices = rng.permutation(
    N_SAMPLES
)

X_true = X_true[indices]
y = y[indices]


# ============================================================
# 5. Phone A
#
# 거의 기준 orientation
# ============================================================

R_A = rotation_matrix_xyz(
    np.deg2rad(5),
    np.deg2rad(-3),
    np.deg2rad(2)
)

X_A = apply_phone_sensor(
    X_true,
    rotation=R_A,
    scale=np.array([
        1.00,
        1.00,
        1.00
    ]),
    bias=np.array([
        0.10,
        -0.05,
        0.08
    ]),
    noise_std=0.06
)


# ============================================================
# 6. Phone B
#
# orientation 크게 다름
# + scale/bias/noise도 다름
# ============================================================

R_B = rotation_matrix_xyz(
    np.deg2rad(35),
    np.deg2rad(-50),
    np.deg2rad(70)
)

X_B = apply_phone_sensor(
    X_true,
    rotation=R_B,
    scale=np.array([
        1.20,
        0.85,
        1.35
    ]),
    bias=np.array([
        -0.40,
        0.25,
        -0.15
    ]),
    noise_std=0.12
)


# ============================================================
# 7. Train / test split
# ============================================================

split = int(
    0.7 * N_SAMPLES
)

train_idx = np.arange(
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
# Feature 1:
# raw xyz statistics
# ============================================================

def raw_xyz_features(X):

    features = []

    for axis in range(3):

        axis_data = X[:, :, axis]

        features.extend([
            axis_data.mean(
                axis=1
            ),
            axis_data.std(
                axis=1
            ),
            axis_data.max(
                axis=1
            ),
            axis_data.min(
                axis=1
            ),
            np.sqrt(
                np.mean(
                    axis_data ** 2,
                    axis=1
                )
            )
        ])

    return np.column_stack(
        features
    )


# ============================================================
# Feature 2:
# axis-wise normalization
#
# 각 축마다 mean/std 제거
#
# bias/scale에는 강해지지만
# orientation rotation은 그대로 남음
# ============================================================

def axis_normalized_features(X):

    mean = X.mean(
        axis=1,
        keepdims=True
    )

    std = X.std(
        axis=1,
        keepdims=True
    )

    Xn = (
        X - mean
    ) / (
        std + 1e-8
    )

    features = []

    for axis in range(3):

        axis_data = Xn[:, :, axis]

        spectrum = np.abs(
            np.fft.rfft(
                axis_data,
                axis=1
            )
        )

        spectrum[:, 0] = 0

        dominant_bin = np.argmax(
            spectrum,
            axis=1
        )

        dominant_power = np.max(
            spectrum,
            axis=1
        )

        zero_crossings = np.sum(
            axis_data[:, 1:]
            * axis_data[:, :-1]
            < 0,
            axis=1
        )

        features.extend([
            dominant_bin,
            dominant_power,
            zero_crossings
        ])

    return np.column_stack(
        features
    )


# ============================================================
# Feature 3:
# rotation-invariant magnitude
#
# ||a|| = sqrt(x^2 + y^2 + z^2)
#
# rotation R에 대해
#
# ||R a|| = ||a||
#
# 따라서 orientation 변화에 강함
# ============================================================

def magnitude_features(X):

    magnitude = np.sqrt(
        np.sum(
            X ** 2,
            axis=2
        )
    )


    # sample-wise normalize
    mean = magnitude.mean(
        axis=1,
        keepdims=True
    )

    std = magnitude.std(
        axis=1,
        keepdims=True
    )

    m = (
        magnitude - mean
    ) / (
        std + 1e-8
    )


    spectrum = np.abs(
        np.fft.rfft(
            m,
            axis=1
        )
    )

    spectrum[:, 0] = 0


    dominant_bin = np.argmax(
        spectrum,
        axis=1
    )

    dominant_power = np.max(
        spectrum,
        axis=1
    )

    zero_crossings = np.sum(
        m[:, 1:]
        * m[:, :-1]
        < 0,
        axis=1
    )


    spectral_centroid = (
        np.sum(
            spectrum
            * np.arange(
                spectrum.shape[1]
            ),
            axis=1
        )
        /
        (
            np.sum(
                spectrum,
                axis=1
            )
            + 1e-8
        )
    )


    return np.column_stack([
        dominant_bin,
        dominant_power,
        zero_crossings,
        spectral_centroid
    ])


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    feature_fn,
    name
):

    train = feature_fn(
        X_A_train
    )

    test_A = feature_fn(
        X_A_test
    )

    test_B = feature_fn(
        X_B_test
    )


    scaler = StandardScaler()

    train = scaler.fit_transform(
        train
    )

    test_A = scaler.transform(
        test_A
    )

    test_B = scaler.transform(
        test_B
    )


    model = LogisticRegression(
        max_iter=2000
    )

    model.fit(
        train,
        y_train
    )


    pred_A = model.predict(
        test_A
    )

    pred_B = model.predict(
        test_B
    )


    acc_A = accuracy_score(
        y_test,
        pred_A
    )

    acc_B = accuracy_score(
        y_test,
        pred_B
    )


    print(name)

    print(
        f"  A → A : "
        f"{acc_A * 100:.2f}%"
    )

    print(
        f"  A → B : "
        f"{acc_B * 100:.2f}%"
    )

    print(
        f"  gap   : "
        f"{(acc_A - acc_B) * 100:.2f}%p"
    )

    print()


# ============================================================
# Run
# ============================================================

print(
    "=== 3-AXIS IMU DOMAIN SHIFT ==="
)

print()

evaluate(
    raw_xyz_features,
    "RAW XYZ"
)

evaluate(
    axis_normalized_features,
    "AXIS-NORMALIZED XYZ"
)

evaluate(
    magnitude_features,
    "ROTATION-INVARIANT MAGNITUDE"
)