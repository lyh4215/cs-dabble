import numpy as np

N = 300
t = np.arange(N)

START = 140
END = 190

x = np.sin(
    2 * np.pi * t / 20
)

x[START:END] = np.sin(
    2 * np.pi
    * t[START:END]
    / 6
)


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


# ==================================================
# 1. coarse detection
# ==================================================

COARSE_WINDOW = 40
COARSE_STRIDE = 20

coarse = []

for start in range(
    0,
    N - COARSE_WINDOW + 1,
    COARSE_STRIDE,
):

    end = start + COARSE_WINDOW

    score = spectral_distance(
        x[start:end]
    )

    coarse.append(
        (start, end, score)
    )


scores = np.array([
    s for _, _, s in coarse
])

threshold = (
    scores.mean()
    + 0.7 * scores.std()
)

detected = [
    (start, end)
    for start, end, score
    in coarse
    if score > threshold
]


coarse_start = min(
    s for s, _ in detected
)

coarse_end = max(
    e for _, e in detected
)


print(
    "actual:",
    (START, END),
)

print(
    "coarse:",
    (
        coarse_start,
        coarse_end,
    ),
)


# ==================================================
# 2. nested / fine detection
#
# coarse region 안에서
# 작은 window로 다시 검사
# ==================================================

FINE_WINDOW = 12
FINE_STRIDE = 2

fine = []

for start in range(
    coarse_start,
    coarse_end - FINE_WINDOW + 1,
    FINE_STRIDE,
):

    end = start + FINE_WINDOW

    score = spectral_distance(
        x[start:end]
    )

    fine.append(
        (start, end, score)
    )


fine_scores = np.array([
    s for _, _, s in fine
])


# fine score는 coarse보다 scale이 달라질 수 있으므로
# fine 내부에서 threshold 새로 계산
fine_threshold = (
    fine_scores.mean()
    + 0.15 * fine_scores.std()
)


fine_detected = [
    (start, end, score)
    for start, end, score
    in fine
    if score > fine_threshold
]


print(
    "\nfine threshold:",
    fine_threshold,
)


for start, end, score in fine_detected:

    print(
        f"[{start:3d}, {end:3d})",
        f"{score:.3f}",
    )


if fine_detected:

    fine_start = min(
        s
        for s, _, _
        in fine_detected
    )

    fine_end = max(
        e
        for _, e, _
        in fine_detected
    )

    print(
        "\nfine detected range:",
        (
            fine_start,
            fine_end,
        ),
    )