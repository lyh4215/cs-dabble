import numpy as np

N = 300
t = np.arange(N)

POINT = 80
START = 140
END = 190

# ==================================================
# 1. signal
# ==================================================

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
#
# median / MAD 기반 robust normalization
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
# 5. combine
# ==================================================

alpha = 0.5

dual = (
    alpha * T
    + (1 - alpha) * F
)


# ==================================================
# 6. compare
# ==================================================

normal_idx = np.r_[
    0:70,
    90:130,
    210:N,
]


print(
    "=== NORMALIZED SCORES ==="
)


print(
    "\nPOINT anomaly"
)

print(
    "time:",
    T[POINT],
)

print(
    "freq:",
    F[POINT],
)

print(
    "dual:",
    dual[POINT],
)


print(
    "\nPATTERN anomaly mean"
)

print(
    "time:",
    T[START:END].mean(),
)

print(
    "freq:",
    F[START:END].mean(),
)

print(
    "dual:",
    dual[START:END].mean(),
)


print(
    "\nNORMAL mean"
)

print(
    "time:",
    T[normal_idx].mean(),
)

print(
    "freq:",
    F[normal_idx].mean(),
)

print(
    "dual:",
    dual[normal_idx].mean(),
)


# ==================================================
# 7. top anomaly locations
# ==================================================

top = np.argsort(
    dual
)[-20:][::-1]


print(
    "\n=== TOP 20 DUAL-SCORE TIMESTAMPS ==="
)

for idx in top:

    label = "normal"

    if idx == POINT:
        label = "POINT"

    elif START <= idx < END:
        label = "PATTERN"

    print(
        f"t={idx:3d}",
        f"dual={dual[idx]:7.2f}",
        label,
    )