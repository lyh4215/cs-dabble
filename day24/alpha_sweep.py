import numpy as np

# dual_score.py에서 만든
# T, F, POINT, START, END, normal_idx
# 그대로 있다고 가정

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
    "alpha | point | pattern | normal"
)

print("-" * 42)

for alpha in np.linspace(
    0,
    1,
    11,
):

    dual = (
        alpha * T
        + (1 - alpha) * F
    )

    point_score = dual[POINT]

    pattern_score = dual[
        START:END
    ].mean()

    normal_score = dual[
        normal_idx
    ].mean()

    print(
        f"{alpha:4.1f} | "
        f"{point_score:6.2f} | "
        f"{pattern_score:7.2f} | "
        f"{normal_score:6.2f}"
    )