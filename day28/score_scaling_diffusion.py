import numpy as np

np.random.seed(1)


# ==================================================
# target pose distribution
#
# two plausible pose modes:
# -2 and +2
#
# 약간 비대칭으로 둬서
# strong guidance의 mode-collapse 경향을 보기 쉽게 함
# ==================================================

MODES = np.array([
    -2.0,
    +2.0,
])

WEIGHTS = np.array([
    0.58,
    0.42,
])


# ==================================================
# score = gradient log p(x)
#
# mixture Gaussian의 score를 정확히 계산
# ==================================================

def score_function(
    x,
    sigma,
):

    # [N, 2]
    log_prob = (
        np.log(
            WEIGHTS
        )[None, :]

        -

        0.5
        * (
            (
                x[:, None]
                - MODES[None, :]
            )
            / sigma
        ) ** 2
    )


    # numerical stability
    m = log_prob.max(
        axis=1,
        keepdims=True,
    )


    responsibility = np.exp(
        log_prob - m
    )

    responsibility /= (
        responsibility.sum(
            axis=1,
            keepdims=True,
        )
    )


    # Gaussian score:
    #
    # (mu - x) / sigma^2
    #
    score = (
        responsibility
        * (
            MODES[None, :]
            - x[:, None]
        )
        / sigma ** 2
    ).sum(
        axis=1
    )


    return score


# ==================================================
# sampler
# ==================================================

def sample(
    strategy,
    seed=1,
):

    rng = np.random.default_rng(
        seed
    )


    N = 5000
    STEPS = 300


    # 처음에는 noise 상태에서 시작
    x = rng.normal(
        0,
        3,
        N,
    )


    snapshots = {}


    for k in range(
        STEPS
    ):

        # diffusion time:
        #
        # 1 → 0
        t = (
            1
            - k
            / (STEPS - 1)
        )


        # early:
        # distribution이 매우 blurred
        #
        # late:
        # 두 pose mode가 뚜렷해짐
        sigma = (
            0.35
            + 2.5 * t
        )


        score = score_function(
            x,
            sigma,
        )


        # ==========================================
        # score scaling
        # ==========================================

        if strategy == "baseline":

            w = 1.0


        elif strategy == "constant":

            # 처음부터 강한 guidance
            w = 6.0


        elif strategy == "schedule":

            w_min = 1.0
            w_max = 6.0

            # paper-style exponential schedule
            w = (
                w_min

                +

                (
                    w_max
                    - w_min
                )
                * np.exp(
                    -5 * t
                )
            )


        else:

            raise ValueError(
                strategy
            )


        # ==========================================
        # toy reverse diffusion update
        # ==========================================

        step = 0.01


        # stochastic exploration
        #
        # early = noise 큼
        # late  = noise 작음
        noise_scale = (
            0.5 * t
            + 0.01
        )


        noise = rng.normal(
            0,
            1,
            N,
        )


        x = (

            x

            +

            step
            * w
            * score

            +

            np.sqrt(
                2 * step
            )
            * noise_scale
            * noise
        )


        # snapshots
        if k in [
            75,
            225,
            299,
        ]:

            distance = np.minimum(
                np.abs(
                    x + 2
                ),

                np.abs(
                    x - 2
                ),
            )


            snapshots[k] = {
                "t": t,
                "w": w,

                "left": (
                    x < 0
                ).mean(),

                "error":
                    distance.mean(),

                "outlier":
                    (
                        distance
                        > 0.3
                    ).mean(),
            }


    return (
        x,
        snapshots,
    )


# ==================================================
# run three strategies
# ==================================================

strategies = [
    "baseline",
    "constant",
    "schedule",
]


for strategy in strategies:

    x, snapshots = sample(
        strategy
    )


    distance = np.minimum(
        np.abs(
            x + 2
        ),

        np.abs(
            x - 2
        ),
    )


    left = (
        x < 0
    ).mean()

    right = 1 - left


    print(
        "\n=============================="
    )

    print(
        strategy.upper()
    )

    print(
        "=============================="
    )


    for k, s in snapshots.items():

        print(
            f"step={k:3d}",
            f"t={s['t']:.2f}",
            f"w={s['w']:.2f}",
            f"left={s['left']:.3f}",
            f"error={s['error']:.3f}",
            f"outlier={s['outlier']:.3f}",
        )


    print(
        "\nFINAL"
    )

    print(
        "left mode :",
        f"{left:.3f}",
    )

    print(
        "right mode:",
        f"{right:.3f}",
    )

    print(
        "mean distance to mode:",
        distance.mean(),
    )

    print(
        "outlier ratio:",
        (
            distance > 0.3
        ).mean(),
    )