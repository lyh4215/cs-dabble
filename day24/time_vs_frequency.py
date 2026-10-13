import numpy as np


N = 300

t = np.arange(N)


# ==================================================
# 1. 정상 신호
#
# period = 20인 sine wave
# ==================================================

x = np.sin(
    2 * np.pi * t / 20
)


# ==================================================
# 2. point anomaly
#
# 한 시점만 크게 튐
# ==================================================

POINT = 80

x[POINT] += 4


# ==================================================
# 3. pattern anomaly
#
# 140~190 동안
# 진동 주기가 20 -> 6으로 변화
# ==================================================

START = 140
END = 190

x[START:END] = np.sin(
    2 * np.pi
    * t[START:END]
    / 6
)


# ==================================================
# Time-domain score
#
# 주변 평균에서 얼마나 벗어났나?
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
# Frequency-domain score
#
# window 안 spectrum이
# 정상 sine spectrum과 얼마나 다른지
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


    # scale 영향 제거
    P = P / (P.sum() + 1e-9)
    Q = Q / (Q.sum() + 1e-9)


    return np.abs(
        P - Q
    ).sum()


# ==================================================
# sliding FFT window
# ==================================================

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
# 결과
# ==================================================

normal_idx = np.r_[
    0:70,
    90:130,
    200:N,
]


print(
    "=== TIME DOMAIN ==="
)

print(
    "point anomaly:",
    time_score[POINT],
)

print(
    "pattern anomaly mean:",
    time_score[
        START:END
    ].mean(),
)

print(
    "normal mean:",
    time_score[
        normal_idx
    ].mean(),
)


print(
    "\n=== FREQUENCY DOMAIN ==="
)

print(
    "point vicinity mean:",
    freq_score[
        POINT-5:POINT+6
    ].mean(),
)

print(
    "pattern anomaly mean:",
    freq_score[
        START:END
    ].mean(),
)

print(
    "normal mean:",
    freq_score[
        normal_idx
    ].mean(),
)