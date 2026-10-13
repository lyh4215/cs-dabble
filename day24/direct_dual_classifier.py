import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# ==================================================
# 1. same data
# ==================================================

N = 300
t = np.arange(N)

POINT = 80
START = 140
END = 190

x = np.sin(
    2 * np.pi * t / 20
)

x[POINT] += 4

x[START:END] = np.sin(
    2 * np.pi
    * t[START:END]
    / 6
)


# ==================================================
# 2. time score
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
# 3. frequency score
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

    return np.abs(
        P - Q
    ).sum()


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
# 4. normalize
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
# 5. labels
# ==================================================

Y = np.zeros(N)

Y[POINT] = 1
Y[START:END] = 1


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
# 6. direct classifier
#
# input = [T, F]
# ==================================================

model = torch.nn.Linear(
    2,
    1,
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.03,
)


# ==================================================
# 7. train
# ==================================================

for step in range(2001):

    logits = model(
        features
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
# 8. inspect
# ==================================================

with torch.no_grad():

    logits = model(
        features
    ).squeeze(1)

    prob = torch.sigmoid(
        logits
    )


print(
    "\n=== LEARNED CLASSIFIER ==="
)

print(
    "weights:",
    model.weight.detach().numpy()
)

print(
    "bias:",
    model.bias.detach().numpy()
)


print(
    "\npoint anomaly prob:",
    prob[POINT].item(),
)

print(
    "pattern anomaly mean prob:",
    prob[START:END].mean().item(),
)

print(
    "t=180 prob:",
    prob[180].item(),
)


normal_idx = np.r_[
    0:70,
    90:130,
    210:N,
]


print(
    "normal mean prob:",
    prob[normal_idx].mean().item(),
)


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
        f"prob={prob[idx].item():.3f}",
    )