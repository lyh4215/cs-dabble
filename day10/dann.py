import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

SEED = 42

N_SAMPLES = 6000
N_TRAIN = 4000

BATCH_SIZE = 128
EPOCHS = 100

DOMAIN_WEIGHT = 3.0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 1. Synthetic domain
#
# feature 0:
# motion feature
#
#   실제 activity와 관련
#   A/B에서 거의 동일
#
#
# feature 1:
# sensor/device feature
#
#   Phone A에서는 label과 강하게 correlated
#
#   하지만 Phone B에서는
#   device shift 때문에 전체 분포가 이동
# ============================================================

def make_domain(
    domain,
    n_samples,
    seed
):

    rng = np.random.default_rng(
        seed
    )


    # 0 = walk
    # 1 = run
    y = rng.integers(
        0,
        2,
        n_samples
    )


    # --------------------------------------------------------
    # Domain-invariant motion information
    #
    # walk ≈ -1.2
    # run  ≈ +1.2
    #
    # 하지만 noise가 커서 완벽히 구분되지는 않음
    # --------------------------------------------------------

    motion = (
        (2 * y - 1)
        * 1.2
        + rng.normal(
            0,
            0.9,
            n_samples
        )
    )


    # --------------------------------------------------------
    # Device-specific sensor information
    #
    # Phone A에서는 아주 쉬운 feature
    #
    # walk ≈ -3
    # run  ≈ +3
    #
    # 그래서 source-only classifier가
    # 이것에 의존하기 쉽다.
    # --------------------------------------------------------

    sensor = (
        (2 * y - 1)
        * 3.0
        + rng.normal(
            0,
            0.7,
            n_samples
        )
    )


    # --------------------------------------------------------
    # Phone B domain shift
    #
    # sensor distribution 전체를 +5 이동
    # --------------------------------------------------------

    if domain == 1:

        sensor += 5.0


    X = np.column_stack([
        motion,
        sensor
    ]).astype(
        np.float32
    )


    return (
        X,
        y.astype(
            np.int64
        )
    )


# Phone A = source domain
X_A, y_A = make_domain(
    domain=0,
    n_samples=N_SAMPLES,
    seed=SEED
)


# Phone B = target domain
X_B, y_B = make_domain(
    domain=1,
    n_samples=N_SAMPLES,
    seed=SEED + 1
)


# ============================================================
# 2. Source-based normalization
#
# 실제 inference 상황처럼
# Phone A train statistics만 사용한다.
# ============================================================

mean_A = X_A[:N_TRAIN].mean(
    axis=0
)

std_A = X_A[:N_TRAIN].std(
    axis=0
) + 1e-6


X_A = (
    X_A - mean_A
) / std_A


X_B = (
    X_B - mean_A
) / std_A


# ============================================================
# 3. Tensor split
# ============================================================

X_A_train = torch.tensor(
    X_A[:N_TRAIN]
)

y_A_train = torch.tensor(
    y_A[:N_TRAIN]
)


X_B_train = torch.tensor(
    X_B[:N_TRAIN]
)


X_A_test = torch.tensor(
    X_A[N_TRAIN:]
)

y_A_test = torch.tensor(
    y_A[N_TRAIN:]
)


X_B_test = torch.tensor(
    X_B[N_TRAIN:]
)

y_B_test = torch.tensor(
    y_B[N_TRAIN:]
)


# ============================================================
# 4. Encoder
#
# input feature
#     ↓
# hidden representation z
#
# DANN의 핵심:
#
# classifier와 domain classifier가
# 같은 z를 사용한다.
# ============================================================

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                2,
                16
            ),
            nn.ReLU(),

            nn.Linear(
                16,
                2
            ),
            nn.ReLU()
        )


    def forward(self, x):

        return self.net(x)


# ============================================================
# 5. Activity classifier
# ============================================================

class ActivityClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(
            2,
            2
        )


    def forward(self, z):

        return self.fc(z)


# ============================================================
# 6. Domain classifier
#
# z를 보고
#
# Phone A인가?
# Phone B인가?
#
# 맞히려고 한다.
# ============================================================

class DomainClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                2,
                16
            ),
            nn.ReLU(),

            nn.Linear(
                16,
                2
            )
        )


    def forward(self, z):

        return self.net(z)


# ============================================================
# 7. Gradient Reversal Layer
#
# Forward:
#
#     GRL(z) = z
#
# 아무것도 안 함.
#
#
# Backward:
#
#     dL/dz
#
# 대신
#
#     -lambda * dL/dz
#
# 로 뒤집는다.
# ============================================================

class GradientReversalFunction(
    Function
):

    @staticmethod
    def forward(
        ctx,
        x,
        lambda_
    ):

        ctx.lambda_ = lambda_

        return x.view_as(x)


    @staticmethod
    def backward(
        ctx,
        grad_output
    ):

        return (
            -ctx.lambda_
            * grad_output,
            None
        )


def gradient_reverse(
    x,
    lambda_
):

    return (
        GradientReversalFunction
        .apply(
            x,
            lambda_
        )
    )


# ============================================================
# Evaluation
# ============================================================

def activity_accuracy(
    encoder,
    classifier,
    X,
    y
):

    encoder.eval()
    classifier.eval()

    with torch.no_grad():

        X = X.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        pred = (
            classifier(
                encoder(X)
            )
            .argmax(
                dim=1
            )
        )


        acc = (
            pred == y
        ).float().mean()


    return acc.item()


# ============================================================
# 8. Baseline
#
# Phone A labeled data만 사용
# ============================================================

def train_baseline():

    set_seed(
        SEED
    )


    encoder = (
        Encoder()
        .to(DEVICE)
    )

    classifier = (
        ActivityClassifier()
        .to(DEVICE)
    )


    optimizer = torch.optim.Adam(
        list(
            encoder.parameters()
        )
        +
        list(
            classifier.parameters()
        ),
        lr=1e-3
    )


    loader = DataLoader(
        TensorDataset(
            X_A_train,
            y_A_train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    for epoch in range(40):

        encoder.train()
        classifier.train()

        for x, y in loader:

            x = x.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )


            optimizer.zero_grad()


            z = encoder(x)

            logits = (
                classifier(z)
            )


            loss = (
                F.cross_entropy(
                    logits,
                    y
                )
            )


            loss.backward()

            optimizer.step()


    return (
        encoder,
        classifier
    )


# ============================================================
# 9. DANN
#
# Source Phone A:
#
#     activity label 있음
#
#
# Target Phone B:
#
#     activity label 사용하지 않음
#
#
# 하지만:
#
#     "이 sample이 A인가 B인가"
#
# 라는 domain label은 알고 있다.
# ============================================================

def train_dann():

    set_seed(
        SEED
    )


    encoder = (
        Encoder()
        .to(DEVICE)
    )

    activity_classifier = (
        ActivityClassifier()
        .to(DEVICE)
    )

    domain_classifier = (
        DomainClassifier()
        .to(DEVICE)
    )


    optimizer = torch.optim.Adam(
        list(
            encoder.parameters()
        )
        +
        list(
            activity_classifier
            .parameters()
        )
        +
        list(
            domain_classifier
            .parameters()
        ),
        lr=1e-3
    )


    source_loader = DataLoader(
        TensorDataset(
            X_A_train,
            y_A_train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )


    # 중요:
    #
    # Phone B의 y label은
    # DANN training에 넣지 않는다.
    target_loader = DataLoader(
        TensorDataset(
            X_B_train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )


    for epoch in range(EPOCHS):

        encoder.train()
        activity_classifier.train()
        domain_classifier.train()


        # ----------------------------------------------------
        # 처음에는 domain pressure를 약하게,
        # 나중에 점점 강하게.
        #
        # DANN에서 자주 사용하는 schedule 형태.
        # ----------------------------------------------------

        progress = (
            epoch
            / (
                EPOCHS - 1
            )
        )


        lambda_ = (
            2.0
            /
            (
                1.0
                + math.exp(
                    -10.0
                    * progress
                )
            )
            - 1.0
        )


        for (
            source_batch,
            target_batch
        ) in zip(
            source_loader,
            target_loader
        ):

            (
                x_source,
                y_source
            ) = source_batch


            (
                x_target,
            ) = target_batch


            x_source = (
                x_source.to(
                    DEVICE
                )
            )

            y_source = (
                y_source.to(
                    DEVICE
                )
            )

            x_target = (
                x_target.to(
                    DEVICE
                )
            )


            optimizer.zero_grad()


            # ------------------------------------------------
            # Encoder
            # ------------------------------------------------

            z_source = (
                encoder(
                    x_source
                )
            )

            z_target = (
                encoder(
                    x_target
                )
            )


            # ------------------------------------------------
            # Activity loss
            #
            # source label만 사용
            # ------------------------------------------------

            activity_logits = (
                activity_classifier(
                    z_source
                )
            )


            activity_loss = (
                F.cross_entropy(
                    activity_logits,
                    y_source
                )
            )


            # ------------------------------------------------
            # Domain loss
            #
            # source = 0
            # target = 1
            # ------------------------------------------------

            z_domain = torch.cat([
                z_source,
                z_target
            ])


            domain_labels = torch.cat([
                torch.zeros(
                    len(z_source),
                    dtype=torch.long,
                    device=DEVICE
                ),

                torch.ones(
                    len(z_target),
                    dtype=torch.long,
                    device=DEVICE
                )
            ])


            # 핵심!
            #
            # domain classifier 앞에
            # gradient reversal을 넣는다.
            reversed_z = (
                gradient_reverse(
                    z_domain,
                    lambda_
                )
            )


            domain_logits = (
                domain_classifier(
                    reversed_z
                )
            )


            domain_loss = (
                F.cross_entropy(
                    domain_logits,
                    domain_labels
                )
            )


            total_loss = (
                activity_loss
                +
                DOMAIN_WEIGHT
                * domain_loss
            )


            total_loss.backward()

            optimizer.step()


    return (
        encoder,
        activity_classifier,
        domain_classifier
    )


# ============================================================
# 10. Domain accuracy
#
# DANN이 정말 domain 정보를 지웠다면
#
# Phone A/B 구분 accuracy가
# 50% 근처로 내려가야 한다.
# ============================================================

def domain_accuracy(
    encoder,
    domain_classifier
):

    encoder.eval()
    domain_classifier.eval()


    with torch.no_grad():

        A = X_A_test.to(
            DEVICE
        )

        B = X_B_test.to(
            DEVICE
        )


        z = torch.cat([
            encoder(A),
            encoder(B)
        ])


        labels = torch.cat([
            torch.zeros(
                len(A),
                dtype=torch.long,
                device=DEVICE
            ),

            torch.ones(
                len(B),
                dtype=torch.long,
                device=DEVICE
            )
        ])


        pred = (
            domain_classifier(z)
            .argmax(
                dim=1
            )
        )


        acc = (
            pred == labels
        ).float().mean()


    return acc.item()


# ============================================================
# 11. Train baseline
# ============================================================

baseline_encoder, baseline_classifier = (
    train_baseline()
)


baseline_A = activity_accuracy(
    baseline_encoder,
    baseline_classifier,
    X_A_test,
    y_A_test
)


baseline_B = activity_accuracy(
    baseline_encoder,
    baseline_classifier,
    X_B_test,
    y_B_test
)


# ============================================================
# 12. Train DANN
# ============================================================

(
    dann_encoder,
    dann_classifier,
    dann_domain_classifier
) = train_dann()


dann_A = activity_accuracy(
    dann_encoder,
    dann_classifier,
    X_A_test,
    y_A_test
)


dann_B = activity_accuracy(
    dann_encoder,
    dann_classifier,
    X_B_test,
    y_B_test
)


dann_domain_acc = (
    domain_accuracy(
        dann_encoder,
        dann_domain_classifier
    )
)


# ============================================================
# 13. Results
# ============================================================

print(
    "=== DOMAIN ADVERSARIAL LEARNING ==="
)

print()

print(
    f"Device: {DEVICE}"
)

print()


print(
    "SOURCE-ONLY BASELINE"
)

print(
    f"  Phone A → A : "
    f"{baseline_A * 100:.2f}%"
)

print(
    f"  Phone A → B : "
    f"{baseline_B * 100:.2f}%"
)

print()


print(
    "DANN"
)

print(
    f"  Phone A → A : "
    f"{dann_A * 100:.2f}%"
)

print(
    f"  Phone A → B : "
    f"{dann_B * 100:.2f}%"
)

print(
    f"  Domain classifier : "
    f"{dann_domain_acc * 100:.2f}%"
)

print()


# ============================================================
# 14. Visualize latent representation
# ============================================================

os.makedirs(
    "dist",
    exist_ok=True
)


dann_encoder.eval()


with torch.no_grad():

    z_A = (
        dann_encoder(
            X_A_test.to(
                DEVICE
            )
        )
        .cpu()
        .numpy()
    )

    z_B = (
        dann_encoder(
            X_B_test.to(
                DEVICE
            )
        )
        .cpu()
        .numpy()
    )


# ------------------------------------------------------------
# plot by domain
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    z_A[:, 0],
    z_A[:, 1],
    s=8,
    alpha=0.35,
    label="Phone A"
)

plt.scatter(
    z_B[:, 0],
    z_B[:, 1],
    s=8,
    alpha=0.35,
    label="Phone B"
)

plt.xlabel("latent z1")
plt.ylabel("latent z2")

plt.title(
    "DANN latent space — domain"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "dist/dann_domain.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# plot by activity
# ------------------------------------------------------------

z_all = np.concatenate([
    z_A,
    z_B
])


y_all = np.concatenate([
    y_A[N_TRAIN:],
    y_B[N_TRAIN:]
])


plt.figure(
    figsize=(7, 6)
)


for label, name in [
    (0, "walk"),
    (1, "run")
]:

    mask = (
        y_all == label
    )


    plt.scatter(
        z_all[mask, 0],
        z_all[mask, 1],
        s=8,
        alpha=0.35,
        label=name
    )


plt.xlabel("latent z1")
plt.ylabel("latent z2")

plt.title(
    "DANN latent space — activity"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "dist/dann_activity.png",
    dpi=160
)

plt.close()


print(
    "Saved:"
)

print(
    "  dist/dann_domain.png"
)

print(
    "  dist/dann_activity.png"
)