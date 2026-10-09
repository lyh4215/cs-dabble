import numpy as np


rng = np.random.default_rng(0)


LAYERS = [
    ("conv", 1.8, 120),
    ("relu", 0.7, 50),
    ("conv", 2.0, 100),
    ("pool", 1.0, 60),
    ("fc",   1.5, 80),
]


signal = []
ground_truth = []

start = 0

for layer, amplitude, length in LAYERS:

    noise = rng.normal(
        0,
        0.20,
        length,
    )

    trace = amplitude + noise

    signal.extend(trace)

    end = start + length

    ground_truth.append(
        (layer, start, end)
    )

    start = end


signal = np.array(signal)


print("=== Ground truth ===")

for layer, start, end in ground_truth:
    print(
        f"{layer:5s}",
        f"{start:3d} -> {end:3d}",
        f"mean={signal[start:end].mean():.3f}",
    )


# --------------------------------------------------
# Simple change-point detector
#
# 좌우 window 평균이 크게 달라지면
# layer boundary라고 판단
# --------------------------------------------------

WINDOW = 15
THRESHOLD = 0.40

scores = []

for i in range(
    WINDOW,
    len(signal) - WINDOW,
):

    left = signal[
        i - WINDOW:i
    ].mean()

    right = signal[
        i:i + WINDOW
    ].mean()

    score = abs(left - right)

    scores.append(
        (score, i)
    )


# 큰 변화 후보부터 선택
scores.sort(reverse=True)


detected = []

for score, pos in scores:

    if score < THRESHOLD:
        continue

    # 같은 boundary 주변에서
    # 중복 검출 방지
    if any(
        abs(pos - x) < WINDOW
        for x in detected
    ):
        continue

    detected.append(pos)


detected.sort()


print("\n=== Detected boundaries ===")

for pos in detected:
    print(pos)


print("\n=== True boundaries ===")

print([
    end
    for _, _, end in ground_truth[:-1]
])