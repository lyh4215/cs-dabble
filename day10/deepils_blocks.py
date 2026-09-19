import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Settings
# ============================================================

SEED = 42

N_TRAIN = 6000
N_TEST = 1500

SEQ_LEN = 200
CHANNELS = 6

BATCH_SIZE = 128
EPOCHS = 10

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_all(SEED)


# ============================================================
# 1. Synthetic 6-axis IMU
#
# input:
#
# [ax ay az gx gy gz] × 200 samples
#
# target:
#
# [vx, vy]
#
#
# Phone B는 같은 motion인데
# sensor scale/bias/noise가 다르다.
# ============================================================

def make_dataset(
    n,
    domain,
    seed
):

    rng = np.random.default_rng(seed)

    X = np.zeros(
        (n, CHANNELS, SEQ_LEN),
        dtype=np.float32
    )

    Y = np.zeros(
        (n, 2),
        dtype=np.float32
    )

    t = np.linspace(
        0,
        2,
        SEQ_LEN
    )


    for i in range(n):

        # --------------------------------------------
        # Ground-truth velocity
        # --------------------------------------------

        speed = rng.uniform(
            0.3,
            2.2
        )

        heading = rng.uniform(
            -np.pi,
            np.pi
        )

        vx = (
            speed
            * np.cos(heading)
        )

        vy = (
            speed
            * np.sin(heading)
        )

        Y[i] = [
            vx,
            vy
        ]


        # --------------------------------------------
        # Walking frequency
        # faster motion -> slightly faster frequency
        # --------------------------------------------

        freq = (
            1.3
            + 0.65 * speed
            + rng.normal(
                0,
                0.08
            )
        )

        phase = rng.uniform(
            0,
            2 * np.pi
        )


        s1 = np.sin(
            2 * np.pi
            * freq
            * t
            + phase
        )

        c1 = np.cos(
            2 * np.pi
            * freq
            * t
            + phase
        )

        s2 = np.sin(
            4 * np.pi
            * freq
            * t
            + 0.4
            + phase
        )


        # --------------------------------------------
        # Synthetic accelerometer
        # --------------------------------------------

        ax = (
            0.80 * vx * s1
            + 0.18 * vy * c1
            + 0.15 * speed * s2
        )

        ay = (
            0.80 * vy * s1
            - 0.18 * vx * c1
            + 0.12 * speed * s2
        )

        az = (
            0.45 * speed * s2
            + 0.10 * s1
        )


        # --------------------------------------------
        # Synthetic gyroscope
        # --------------------------------------------

        gx = (
            0.20 * vy * c1
            + 0.10 * speed * s2
        )

        gy = (
            -0.20 * vx * c1
            + 0.08 * speed * s2
        )

        gz = (
            0.18
            * (vx - vy)
            * s1
            + 0.10 * heading * c1
        )


        signal = np.stack([
            ax,
            ay,
            az,
            gx,
            gy,
            gz
        ])


        # 사람 움직임 자체 noise
        signal += rng.normal(
            0,
            0.05,
            signal.shape
        )


        # --------------------------------------------
        # Device domain
        # --------------------------------------------

        if domain == "A":

            scale = np.array([
                1.00,
                1.00,
                1.00,
                1.00,
                1.00,
                1.00
            ])[:, None]

            bias = np.array([
                0.04,
                -0.03,
                0.02,
                0.01,
                -0.01,
                0.02
            ])[:, None]

            noise = 0.04


        else:

            scale = np.array([
                1.12,
                0.91,
                1.08,
                0.88,
                1.15,
                1.05
            ])[:, None]

            bias = np.array([
                -0.12,
                0.10,
                -0.08,
                0.06,
                -0.05,
                0.08
            ])[:, None]

            noise = 0.10


        signal = (
            signal * scale
            + bias
            + rng.normal(
                0,
                noise,
                signal.shape
            )
        )


        X[i] = signal


    return X, Y


# ============================================================
# Data
# ============================================================

X_train, y_train = make_dataset(
    N_TRAIN,
    "A",
    SEED
)

X_test_A, y_test_A = make_dataset(
    N_TEST,
    "A",
    SEED + 1
)

X_test_B, y_test_B = make_dataset(
    N_TEST,
    "B",
    SEED + 2
)


# ============================================================
# Per-window normalization
#
# sensor bias/scale 차이를 어느 정도 감소
# ============================================================

def normalize(X):

    mean = X.mean(
        axis=2,
        keepdims=True
    )

    std = X.std(
        axis=2,
        keepdims=True
    )

    return (
        (X - mean)
        /
        (std + 1e-6)
    ).astype(
        np.float32
    )


X_train = normalize(X_train)
X_test_A = normalize(X_test_A)
X_test_B = normalize(X_test_B)


train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 2. Plain convolution residual block
# ============================================================

class PlainResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )


        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()


    def forward(self, x):

        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual

        return torch.relu(x)


# ============================================================
# 3. Depthwise-separable convolution
#
# depthwise:
#   channel마다 temporal convolution
#
# pointwise:
#   channel 간 mixing
# ============================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )


    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


# ============================================================
# 4. Channel Attention
#
# 질문:
#
# "어떤 learned channel이 중요한가?"
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        hidden = max(
            channels // reduction,
            4
        )

        self.mlp = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden,
                kernel_size=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                hidden,
                channels,
                kernel_size=1
            )
        )


    def forward(self, x):

        avg = torch.mean(
            x,
            dim=2,
            keepdim=True
        )

        maximum = torch.amax(
            x,
            dim=2,
            keepdim=True
        )

        attention = torch.sigmoid(
            self.mlp(avg)
            +
            self.mlp(maximum)
        )

        return x * attention


# ============================================================
# 5. Spatial / Temporal Attention
#
# 1D IMU에서는 사실상:
#
# "어느 시간 구간이 중요한가?"
# ============================================================

class TemporalAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv1d(
            2,
            1,
            kernel_size=7,
            padding=3
        )


    def forward(self, x):

        avg = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        maximum = torch.amax(
            x,
            dim=1,
            keepdim=True
        )

        pooled = torch.cat([
            avg,
            maximum
        ], dim=1)

        attention = torch.sigmoid(
            self.conv(pooled)
        )

        return x * attention


# ============================================================
# 6. DWS residual block
# ============================================================

class DWSResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        attention=False
    ):

        super().__init__()

        self.conv1 = (
            DepthwiseSeparableConv(
                in_channels,
                out_channels,
                stride
            )
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.conv2 = (
            DepthwiseSeparableConv(
                out_channels,
                out_channels
            )
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )


        if attention:

            self.channel_attention = (
                ChannelAttention(
                    out_channels
                )
            )

            self.temporal_attention = (
                TemporalAttention()
            )

        else:

            self.channel_attention = (
                nn.Identity()
            )

            self.temporal_attention = (
                nn.Identity()
            )


        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()


    def forward(self, x):

        residual = self.shortcut(x)


        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)


        x = self.conv2(x)
        x = self.bn2(x)


        x = self.channel_attention(x)

        x = self.temporal_attention(x)


        x = x + residual

        return torch.relu(x)


# ============================================================
# 7. Whole velocity network
# ============================================================

class VelocityNet(nn.Module):

    def __init__(
        self,
        block_type
    ):

        super().__init__()


        self.stem = nn.Sequential(
            nn.Conv1d(
                6,
                32,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),

            nn.BatchNorm1d(
                32
            ),

            nn.ReLU()
        )


        if block_type == "plain":

            block = PlainResidualBlock

            self.blocks = nn.Sequential(
                block(32, 64, 2),
                block(64, 128, 2),
                block(128, 128, 2)
            )


        elif block_type == "dws":

            self.blocks = nn.Sequential(
                DWSResidualBlock(
                    32,
                    64,
                    2,
                    False
                ),

                DWSResidualBlock(
                    64,
                    128,
                    2,
                    False
                ),

                DWSResidualBlock(
                    128,
                    128,
                    2,
                    False
                )
            )


        elif block_type == "attention":

            self.blocks = nn.Sequential(
                DWSResidualBlock(
                    32,
                    64,
                    2,
                    True
                ),

                DWSResidualBlock(
                    64,
                    128,
                    2,
                    True
                ),

                DWSResidualBlock(
                    128,
                    128,
                    2,
                    True
                )
            )


        else:

            raise ValueError(
                block_type
            )


        self.pool = (
            nn.AdaptiveAvgPool1d(1)
        )


        self.output = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.stem(x)

        x = self.blocks(x)

        x = self.pool(x)

        return self.output(x)


# ============================================================
# 8. Metrics
# ============================================================

def parameter_count(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def rmse(
    model,
    X,
    y
):

    model.eval()

    with torch.no_grad():

        X = torch.tensor(
            X,
            device=DEVICE
        )

        y = torch.tensor(
            y,
            device=DEVICE
        )

        pred = model(X)

        mse = torch.mean(
            (pred - y) ** 2
        )

    return (
        torch.sqrt(mse)
        .item()
    )


# ============================================================
# 9. Training
# ============================================================

def train_model(kind):

    seed_all(SEED)

    model = (
        VelocityNet(kind)
        .to(DEVICE)
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-3
    )


    for epoch in range(EPOCHS):

        model.train()

        running = 0.0


        for X, y in train_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)


            optimizer.zero_grad()


            pred = model(X)

            loss = torch.mean(
                (pred - y) ** 2
            )


            loss.backward()

            optimizer.step()


            running += (
                loss.item()
                * len(X)
            )


        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
        ):

            print(
                f"  epoch "
                f"{epoch + 1:2d}/{EPOCHS}"
                f"  mse="
                f"{running / N_TRAIN:.5f}"
            )


    return model


# ============================================================
# 10. Latency benchmark
# ============================================================

def benchmark_latency(
    model,
    batch_size=1,
    runs=300
):

    model.eval()


    x = torch.randn(
        batch_size,
        6,
        SEQ_LEN,
        device=DEVICE
    )


    with torch.inference_mode():

        # warmup
        for _ in range(50):
            model(x)


        if DEVICE == "cuda":
            torch.cuda.synchronize()


        start = time.perf_counter()


        for _ in range(runs):
            model(x)


        if DEVICE == "cuda":
            torch.cuda.synchronize()


        end = time.perf_counter()


    return (
        (end - start)
        / runs
        * 1000
    )


# ============================================================
# 11. Run all models
# ============================================================

configs = [
    ("plain", "PLAIN CONV"),
    ("dws", "DEPTHWISE SEPARABLE"),
    (
        "attention",
        "DWS + CHANNEL/SPATIAL ATTENTION"
    )
]


results = []


print(
    "=== DEEPILS BLOCK EXPERIMENT ==="
)

print(
    f"Device: {DEVICE}"
)

print()


for kind, name in configs:

    print(
        f"[{name}]"
    )


    model = train_model(
        kind
    )


    params = parameter_count(
        model
    )


    test_A = rmse(
        model,
        X_test_A,
        y_test_A
    )

    test_B = rmse(
        model,
        X_test_B,
        y_test_B
    )


    latency_1 = benchmark_latency(
        model,
        batch_size=1
    )


    latency_64 = benchmark_latency(
        model,
        batch_size=64,
        runs=100
    )


    results.append(
        (
            name,
            params,
            test_A,
            test_B,
            latency_1,
            latency_64
        )
    )


    print()


# ============================================================
# 12. Summary
# ============================================================

print()
print(
    "=== SUMMARY ==="
)

print(
    f"{'model':<34}"
    f"{'params':>11}"
    f"{'A RMSE':>11}"
    f"{'B RMSE':>11}"
    f"{'B1 ms':>11}"
    f"{'B64 ms':>11}"
)

print("-" * 89)


for (
    name,
    params,
    test_A,
    test_B,
    latency_1,
    latency_64
) in results:

    print(
        f"{name:<34}"
        f"{params:>11,}"
        f"{test_A:>11.4f}"
        f"{test_B:>11.4f}"
        f"{latency_1:>11.3f}"
        f"{latency_64:>11.3f}"
    )