import numpy as np

rng = np.random.default_rng(0)

d = 16
rank = 4

A = rng.normal(size=(rank, d))
B = rng.normal(size=(d, rank))

delta = B @ A

# singular values
_, s, _ = np.linalg.svd(delta)

sigma_max = s[0]

print("singular values:")
print(np.round(s, 3))

print("\nsigma_max:", sigma_max)


# random input들을 넣어 실제 amplification 측정
max_ratio = 0.0

for _ in range(100_000):

    x = rng.normal(size=d)

    ratio = (
        np.linalg.norm(delta @ x)
        / np.linalg.norm(x)
    )

    max_ratio = max(max_ratio, ratio)


print("largest observed ||ΔW x|| / ||x||:")
print(max_ratio)


# spectral clipping
tau = 5.0

U, s, Vt = np.linalg.svd(delta)

s_clipped = np.minimum(s, tau)

delta_clipped = (
    U @ np.diag(s_clipped) @ Vt
)

print("\nafter clipping")
print("singular values:")
print(np.round(s_clipped, 3))

print(
    "spectral norm:",
    np.linalg.norm(delta_clipped, ord=2)
)