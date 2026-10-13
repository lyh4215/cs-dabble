import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)


# ==================================================
# utilities
# ==================================================

def rolling_mean(x, window):

    pad = window // 2

    padded = np.pad(
        x,
        (pad, pad),
        mode="reflect",
    )

    kernel = np.ones(window) / window

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def local_stat(
    x,
    window,
    mode,
):

    pad = window // 2

    padded = np.pad(
        x,
        (pad, pad),
        mode="reflect",
    )

    result = []

    for i in range(len(x)):

        seg = padded[
            i:i + window
        ]

        if mode == "mean":
            result.append(
                seg.mean()
            )

        else:
            result.append(
                seg.std()
            )

    return np.array(
        result
    )


def robust_normalize(x):

    med = np.median(x)

    mad = np.median(
        np.abs(x - med)
    )

    z = (
        x - med
    ) / (
        mad + 1e-6
    )

    return np.maximum(
        z,
        0,
    )


# ==================================================
# one synthetic time series
# ==================================================

def make_sequence(
    N=300,
):

    t = np.arange(N)

    # ------------------------------------------------
    # normal signal
    # ------------------------------------------------

    base_period = np.random.uniform(
        18,
        22,
    )

    phase = np.random.uniform(
        0,
        2 * np.pi,
    )

    x = np.sin(
        2 * np.pi * t
        / base_period
        + phase
    )

    x += np.random.normal(
        0,
        0.04,
        N,
    )


    # ------------------------------------------------
    # pattern anomaly
    #
    # 원래 frequency +
    # 새로운 high-frequency 성분 추가
    #
    # 값 자체는 크게 튀지 않도록 설계
    # ------------------------------------------------

    start = np.random.randint(
        120,
        190,
    )

    length = np.random.randint(
        35,
        55,
    )

    end = min(
        start + length,
        260,
    )

    new_period = np.random.uniform(
        4.5,
        7,
    )


    x[start:end] = (

        0.65
        * np.sin(
            2 * np.pi
            * t[start:end]
            / base_period
            + phase
        )

        +

        0.35
        * np.sin(
            2 * np.pi
            * t[start:end]
            / new_period
            + phase
        )

        +

        np.random.normal(
            0,
            0.04,
            end - start,
        )
    )


    # ------------------------------------------------
    # point anomaly
    # ------------------------------------------------

    candidates = (
        list(range(90, 115))
        +
        list(range(270, 290))
    )

    point = np.random.choice(
        candidates
    )

    x[point] += (
        np.random.choice(
            [-1, +1]
        )

        * np.random.uniform(
            3,
            4,
        )
    )


    # ==================================================
    # labels
    #
    # 0 = normal
    # 1 = anomaly
    #
    # type:
    # 0 normal
    # 1 point
    # 2 pattern
    # ==================================================

    y = np.zeros(
        N,
        dtype=np.float32,
    )

    anomaly_type = np.zeros(
        N,
        dtype=np.int64,
    )


    y[start:end] = 1
    anomaly_type[start:end] = 2

    y[point] = 1
    anomaly_type[point] = 1


    # ==================================================
    # Time score
    # ==================================================

    local_mean = rolling_mean(
        x,
        11,
    )

    T = np.abs(
        x - local_mean
    )

    T = robust_normalize(
        T
    )


    # ==================================================
    # Frequency score
    #
    # 초반 정상 window들로
    # reference spectrum 생성
    # ==================================================

    WINDOW = 32

    reference_spectra = []

    for s in [
        0,
        16,
        32,
        48,
    ]:

        seg = x[
            s:s + WINDOW
        ]

        P = np.abs(
            np.fft.rfft(seg)
        ) ** 2

        P /= (
            P.sum() + 1e-9
        )

        reference_spectra.append(
            P
        )


    reference = np.mean(
        reference_spectra,
        axis=0,
    )


    freq_score = np.zeros(N)
    count = np.zeros(N)


    for s in range(
        N - WINDOW + 1
    ):

        seg = x[
            s:s + WINDOW
        ]

        P = np.abs(
            np.fft.rfft(seg)
        ) ** 2

        P /= (
            P.sum() + 1e-9
        )


        distance = np.abs(
            P - reference
        ).sum()


        freq_score[
            s:s + WINDOW
        ] += distance

        count[
            s:s + WINDOW
        ] += 1


    F = (
        freq_score
        / np.maximum(
            count,
            1,
        )
    )

    F = robust_normalize(
        F
    )


    # ==================================================
    # contextual features
    #
    # Gate가 현재 한 점만 보지 않고
    # 주변 상황까지 보게 한다.
    # ==================================================

    mean_T = local_stat(
        T,
        15,
        "mean",
    )

    mean_F = local_stat(
        F,
        15,
        "mean",
    )

    std_T = local_stat(
        T,
        15,
        "std",
    )

    std_F = local_stat(
        F,
        15,
        "std",
    )


    derivative = np.r_[
        0,
        np.abs(
            np.diff(x)
        ),
    ]


    local_signal_std = local_stat(
        x,
        15,
        "std",
    )


    features = np.stack(
        [
            T,
            F,

            mean_T,
            mean_F,

            std_T,
            std_F,

            derivative,
            local_signal_std,
        ],
        axis=1,
    ).astype(
        np.float32
    )


    return (
        features,
        y,
        anomaly_type,
    )


# ==================================================
# Dataset
# ==================================================

def build_dataset(
    num_sequences,
):

    sequences = [
        make_sequence()
        for _ in range(
            num_sequences
        )
    ]


    X = np.concatenate([
        s[0]
        for s in sequences
    ])


    Y = np.concatenate([
        s[1]
        for s in sequences
    ])


    TYPE = np.concatenate([
        s[2]
        for s in sequences
    ])


    return (
        torch.tensor(X),
        torch.tensor(Y),
        torch.tensor(TYPE),
    )


print(
    "building dataset..."
)


X_train, Y_train, TYPE_train = (
    build_dataset(200)
)

X_test, Y_test, TYPE_test = (
    build_dataset(80)
)


print(
    "train:",
    X_train.shape,
)

print(
    "test:",
    X_test.shape,
)


# ==================================================
# class imbalance weight
# ==================================================

positive = Y_train.sum()

negative = (
    len(Y_train)
    - positive
)

pos_weight = torch.tensor([
    float(
        negative
        / positive
    )
])


# ==================================================
# simple expert trainer
# ==================================================

def train_expert(
    columns,
    steps=1000,
):

    model = torch.nn.Linear(
        len(columns),
        1,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.03,
    )


    for _ in range(
        steps
    ):

        logits = model(
            X_train[:, columns]
        ).squeeze(1)


        loss = (
            torch.nn.functional
            .binary_cross_entropy_with_logits(
                logits,
                Y_train,
                pos_weight=pos_weight,
            )
        )


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return model


# ==================================================
# 1. Time-only expert
#
# intentionally only T
# ==================================================

time_expert = train_expert(
    [0]
)


# ==================================================
# 2. Frequency-only expert
# ==================================================

freq_expert = train_expert(
    [1]
)


# ==================================================
# 3. Fixed dual classifier
#
# global fixed weights
#
# z = wT*T + wF*F + b
# ==================================================

fixed_dual = train_expert(
    [0, 1]
)


# ==================================================
# 4. freeze experts
#
# 이제 gate가 expert 자체를
# 이상하게 바꾸지 못하게 한다.
# ==================================================

for p in time_expert.parameters():
    p.requires_grad = False

for p in freq_expert.parameters():
    p.requires_grad = False


# ==================================================
# Context-dependent gate
# ==================================================

class ContextGate(
    torch.nn.Module
):

    def __init__(self):

        super().__init__()

        self.net = (
            torch.nn.Sequential(

                torch.nn.Linear(
                    8,
                    16,
                ),

                torch.nn.ReLU(),

                torch.nn.Linear(
                    16,
                    1,
                ),

                torch.nn.Sigmoid(),
            )
        )


    def forward(
        self,
        X,
    ):

        return (
            self.net(X)
            .squeeze(1)
        )


gate = ContextGate()


# ==================================================
# precompute expert logits
# ==================================================

with torch.no_grad():

    z_time_train = (
        time_expert(
            X_train[:, [0]]
        )
        .squeeze(1)
    )


    z_freq_train = (
        freq_expert(
            X_train[:, [1]]
        )
        .squeeze(1)
    )


optimizer = torch.optim.Adam(
    gate.parameters(),
    lr=0.01,
)


# ==================================================
# train adaptive gate
# ==================================================

for step in range(
    1201
):

    alpha = gate(
        X_train
    )


    final_logit = (

        alpha
        * z_time_train

        +

        (1 - alpha)
        * z_freq_train
    )


    loss = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            final_logit,
            Y_train,
            pos_weight=pos_weight,
        )
    )


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if step % 300 == 0:

        print(
            f"gate step={step:4d}",
            f"loss={loss.item():.4f}",
        )


# ==================================================
# evaluation
# ==================================================

def metrics(
    probabilities,
    name,
):

    pred = (
        probabilities > 0.5
    )

    truth = (
        Y_test > 0.5
    )


    TP = (
        pred & truth
    ).sum().item()

    FP = (
        pred & ~truth
    ).sum().item()

    FN = (
        ~pred & truth
    ).sum().item()


    precision = (
        TP
        / (TP + FP + 1e-9)
    )

    recall = (
        TP
        / (TP + FN + 1e-9)
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-9
        )
    )


    accuracy = (
        pred == truth
    ).float().mean().item()


    point_recall = (
        pred[
            TYPE_test == 1
        ]
        .float()
        .mean()
        .item()
    )


    pattern_recall = (
        pred[
            TYPE_test == 2
        ]
        .float()
        .mean()
        .item()
    )


    print(
        f"{name:12s}",
        f"acc={accuracy:.2%}",
        f"F1={f1:.3f}",
        f"point={point_recall:.2%}",
        f"pattern={pattern_recall:.2%}",
    )


with torch.no_grad():

    # time
    P_time = torch.sigmoid(
        time_expert(
            X_test[:, [0]]
        ).squeeze(1)
    )


    # frequency
    P_freq = torch.sigmoid(
        freq_expert(
            X_test[:, [1]]
        ).squeeze(1)
    )


    # fixed global weights
    P_fixed = torch.sigmoid(
        fixed_dual(
            X_test[:, [0, 1]]
        ).squeeze(1)
    )


    # adaptive
    z_time = (
        time_expert(
            X_test[:, [0]]
        )
        .squeeze(1)
    )


    z_freq = (
        freq_expert(
            X_test[:, [1]]
        )
        .squeeze(1)
    )


    alpha = gate(
        X_test
    )


    z_adaptive = (

        alpha * z_time

        +

        (1 - alpha)
        * z_freq
    )


    P_adaptive = torch.sigmoid(
        z_adaptive
    )


print(
    "\n=== TEST RESULTS ==="
)

metrics(
    P_time,
    "Time only",
)

metrics(
    P_freq,
    "Freq only",
)

metrics(
    P_fixed,
    "Fixed dual",
)

metrics(
    P_adaptive,
    "Adaptive",
)


# ==================================================
# inspect learned alpha
# ==================================================

print(
    "\n=== ADAPTIVE WEIGHTS ==="
)


print(
    "point anomaly alpha:",
    alpha[
        TYPE_test == 1
    ].mean().item(),
)


print(
    "pattern anomaly alpha:",
    alpha[
        TYPE_test == 2
    ].mean().item(),
)


print(
    "normal alpha:",
    alpha[
        TYPE_test == 0
    ].mean().item(),
)