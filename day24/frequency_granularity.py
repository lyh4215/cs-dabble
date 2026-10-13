import numpy as np

N = 300
t = np.arange(N)

START = 140
END = 190

# --------------------------------------------------
# normal signal
# --------------------------------------------------

x = np.sin(
    2 * np.pi * t / 20
)

# --------------------------------------------------
# anomaly:
# 주기 20 -> 6
# --------------------------------------------------

x[START:END] = np.sin(
    2 * np.pi
    * t[START:END]
    / 6
)


# --------------------------------------------------
# spectral distance
# --------------------------------------------------

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


# --------------------------------------------------
# coarse frequency detector
#
# window 하나마다 score 하나만 계산
# --------------------------------------------------

WINDOW = 40
STRIDE = 20

windows = []

for start in range(
    0,
    N - WINDOW + 1,
    STRIDE
):

    end = start + WINDOW

    score = spectral_distance(
        x[start:end]
    )

    windows.append(
        (
            start,
            end,
            score,
        )
    )


# --------------------------------------------------
# 출력
# --------------------------------------------------

print(
    "actual anomaly:",
    f"[{START}, {END})"
)

print(
    "\n=== frequency windows ==="
)

for start, end, score in windows:

    marker = ""

    # anomaly와 조금이라도 겹치면 표시
    if (
        start < END
        and end > START
    ):
        marker = "<-- overlaps anomaly"

    print(
        f"[{start:3d}, {end:3d})",
        f"score={score:.3f}",
        marker
    )


# --------------------------------------------------
# threshold로 anomaly window 선택
# --------------------------------------------------

scores = np.array([
    s for _, _, s in windows
])

threshold = (
    scores.mean()
    + 0.7 * scores.std()
)

detected = [
    (start, end)
    for start, end, score
    in windows
    if score > threshold
]


print(
    "\nthreshold:",
    threshold
)

print(
    "detected windows:",
    detected
)


if detected:

    detected_start = min(
        x[0]
        for x in detected
    )

    detected_end = max(
        x[1]
        for x in detected
    )

    print(
        "coarse detected range:",
        (
            detected_start,
            detected_end,
        )
    )