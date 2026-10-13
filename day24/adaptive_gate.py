import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# ==================================================
# 1. same synthetic signal
# ==================================================

N = 300
t = np.arange(N)

POINT = 80
START = 140
END = 190

x = np.sin(
    2 * np.pi * t / 20
)

# point anomaly
x[POINT] += 4

# pattern anomaly
x[START:END] = np.sin(
    2 * np.pi
    * t[START:END]
    / 6
)


# ==================================================
# 2. time-domain score
# ==================================================

def rolling_mean(x, window=11):

    pad = window // 2

    padded = np.pad(
        x,
        (pad, pad),
        mode="reflect",
    )

    kernel = (
        np.ones(window)
        / window
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


local_mean = rolling_mean(
    x,
    11,
)

time_score = np.abs(
    x - local_mean
)


# ==================================================
# 3. frequency-domain score
# ==================================================

def spectral_distance(segment):

    L = len(segment)

    tt = np.arange(L)

    normal = np.sin(
        2 * np.pi * tt / 20
    )

    P = np.abs(
        np.fft.rfft(segment)
    ) ** 2

    Q = np.abs(
        np.fft.rfft(normal)
    ) ** 2

    P /= P.sum() + 1e-9
    Q /= Q.sum() + 1e-9

    return np.abs(P - Q).sum()


WINDOW = 32

freq_score = np.zeros(N)
count = np.zeros(N)

for start in range(
    N - WINDOW + 1
):

    end = start + WINDOW

    score = spectral_distance(
        x[start:end]
    )

    freq_score[start:end] += score
    count[start:end] += 1


freq_score /= np.maximum(
    count,
    1,
)


# ==================================================
# 4. robust normalization
# ==================================================

def robust_normalize(score):

    med = np.median(score)

    mad = np.median(
        np.abs(score - med)
    )

    z = (
        score - med
    ) / (
        mad + 1e-6
    )

    return np.maximum(
        z,
        0,
    )


T = robust_normalize(
    time_score
)

F = robust_normalize(
    freq_score
)


# ==================================================
# 5. anomaly labels
#
# gate에게 point/pattern 종류는 안 알려줌.
#
# normal  = 0
# anomaly = 1
# ==================================================

Y = np.zeros(N)

Y[POINT] = 1
Y[START:END] = 1


# ==================================================
# 6. Torch tensors
# ==================================================

features = torch.tensor(
    np.stack(
        [T, F],
        axis=1,
    ),
    dtype=torch.float32,
)

labels = torch.tensor(
    Y,
    dtype=torch.float32,
)


# ==================================================
# 7. Adaptive gate
#
# input:
# [T_t, F_t]
#
# output:
# alpha_t ∈ [0,1]
#
# alpha → 1 : time 위주
# alpha → 0 : freq 위주
# ==================================================

gate = torch.nn.Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
    torch.nn.Sigmoid(),
)


# fused score를 실제 anomaly probability로 바꾸는
# 작은 classifier
classifier = torch.nn.Linear(
    1,
    1,
)


optimizer = torch.optim.Adam(
    list(gate.parameters())
    + list(classifier.parameters()),
    lr=0.03,
)


# ==================================================
# 8. training
# ==================================================

for step in range(2001):

    alpha = gate(
        features
    ).squeeze(1)


    T_tensor = features[:, 0]
    F_tensor = features[:, 1]


    fused = (
        alpha * T_tensor
        + (1 - alpha) * F_tensor
    )


    logits = classifier(
        fused[:, None]
    ).squeeze(1)


    loss = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            logits,
            labels,
        )
    )


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if step % 400 == 0:

        with torch.no_grad():

            prob = torch.sigmoid(
                logits
            )

            pred = (
                prob > 0.5
            ).float()

            acc = (
                pred == labels
            ).float().mean()

            print(
                f"step={step:4d}",
                f"loss={loss.item():.4f}",
                f"acc={acc.item():.2%}",
            )


# ==================================================
# 9. inspect learned alpha
# ==================================================

with torch.no_grad():

    alpha = gate(
        features
    ).squeeze(1)

    fused = (
        alpha * features[:, 0]
        + (1 - alpha) * features[:, 1]
    )

    logits = classifier(
        fused[:, None]
    ).squeeze(1)

    prob = torch.sigmoid(
        logits
    )


print(
    "\n=== LEARNED ALPHA ==="
)

print(
    "point anomaly:"
)

print(
    " T =",
    T[POINT],
)

print(
    " F =",
    F[POINT],
)

print(
    " alpha =",
    alpha[POINT].item(),
)

print(
    " fused =",
    fused[POINT].item(),
)

print(
    " anomaly prob =",
    prob[POINT].item(),
)


print(
    "\npattern anomaly mean:"
)

print(
    " T =",
    T[START:END].mean(),
)

print(
    " F =",
    F[START:END].mean(),
)

print(
    " alpha =",
    alpha[START:END]
    .mean()
    .item(),
)

print(
    " fused =",
    fused[START:END]
    .mean()
    .item(),
)

print(
    " anomaly prob =",
    prob[START:END]
    .mean()
    .item(),
)


normal_idx = np.r_[
    0:70,
    90:130,
    210:N,
]


print(
    "\nnormal mean:"
)

print(
    "alpha =",
    alpha[normal_idx]
    .mean()
    .item(),
)

print(
    "prob =",
    prob[normal_idx]
    .mean()
    .item(),
)


# ==================================================
# 10. inspect several timestamps
# ==================================================

print(
    "\n=== SAMPLE TIMESTAMPS ==="
)

for idx in [
    20,
    80,
    145,
    160,
    180,
    230,
]:

    print(
        f"t={idx:3d}",
        f"T={T[idx]:6.2f}",
        f"F={F[idx]:6.2f}",
        f"alpha={alpha[idx].item():.3f}",
        f"prob={prob[idx].item():.3f}",
    )