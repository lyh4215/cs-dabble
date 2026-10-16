import numpy as np
import torch


torch.manual_seed(0)
np.random.seed(0)


# ==================================================
# 1. Diffusion schedule
# ==================================================

T = 100

betas = torch.linspace(
    1e-4,
    0.02,
    T,
)

alphas = (
    1.0
    - betas
)

alpha_bar = torch.cumprod(
    alphas,
    dim=0,
)


# ==================================================
# 2. Noise prediction network
#
# input:
#   x_t
#   normalized timestep tau
#
# output:
#   predicted noise epsilon
# ==================================================

class NoisePredictor(
    torch.nn.Module
):

    def __init__(self):

        super().__init__()

        self.net = torch.nn.Sequential(

            # input:
            # [x_t, tau]
            #
            # 1 + 1 = 2 dimensions

            torch.nn.Linear(
                2,
                64,
            ),

            torch.nn.SiLU(),

            torch.nn.Linear(
                64,
                64,
            ),

            torch.nn.SiLU(),

            torch.nn.Linear(
                64,
                1,
            ),
        )


    def forward(
        self,
        x_t,
        t,
    ):

        # ------------------------------------------
        # timestep:
        #
        # 0 ~ 99
        #   ↓
        # 0.0 ~ 1.0
        # ------------------------------------------

        tau = (
            t.float()
            / (T - 1)
        )


        # shape:
        #
        # tau
        # [BATCH]
        #
        # →
        #
        # [BATCH, 1]

        tau = tau[:, None]


        # ------------------------------------------
        # concatenate
        #
        # x_t : [BATCH, 1]
        # tau : [BATCH, 1]
        #
        # result:
        # [BATCH, 2]
        # ------------------------------------------

        inp = torch.cat(
            [
                x_t,
                tau,
            ],
            dim=1,
        )


        return self.net(
            inp
        )


model = NoisePredictor()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-3,
)


# ==================================================
# 3. Training
#
# clean data distribution:
#
# x0 = -2
# or
# x0 = +2
#
# 해석:
# -2 ≈ 0 degree pose
# +2 ≈ 180 degree pose
# ==================================================

STEPS = 3000
BATCH = 512


for step in range(
    STEPS + 1
):

    # ==============================================
    # 3-1. Sample clean data x0
    # ==============================================

    left_or_right = (
        torch.rand(
            BATCH,
            1,
        )
        < 0.5
    )


    x0 = torch.where(

        left_or_right,

        torch.full(
            (BATCH, 1),
            -2.0,
        ),

        torch.full(
            (BATCH, 1),
            +2.0,
        ),
    )


    # ==============================================
    # 3-2. Random timestep for each sample
    #
    # sample마다 서로 다른 noise level
    # ==============================================

    t = torch.randint(
        0,
        T,
        (BATCH,),
    )


    # ==============================================
    # 3-3. Gaussian noise
    #
    # epsilon ~ N(0, 1)
    # ==============================================

    epsilon = torch.randn_like(
        x0
    )


    # ==============================================
    # 3-4. Forward diffusion
    #
    # x_t =
    #
    # sqrt(alpha_bar_t) * x0
    #
    # +
    #
    # sqrt(1-alpha_bar_t) * epsilon
    #
    # x0 → xt를 한 번에 계산
    # ==============================================

    a_bar = alpha_bar[
        t
    ][:, None]


    x_t = (

        torch.sqrt(
            a_bar
        )
        * x0

        +

        torch.sqrt(
            1 - a_bar
        )
        * epsilon
    )


    # ==============================================
    # 3-5. Predict noise
    #
    # model receives:
    #
    # x_t
    # normalized timestep tau
    #
    # and tries to predict epsilon
    # ==============================================

    predicted_epsilon = model(
        x_t,
        t,
    )


    # ==============================================
    # 3-6. Noise prediction loss
    #
    # actual noise epsilon
    # vs
    # predicted epsilon
    # ==============================================

    loss = (
        (
            predicted_epsilon
            - epsilon
        ) ** 2
    ).mean()


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if step % 500 == 0:

        print(
            f"step={step:4d}",
            f"loss={loss.item():.4f}",
        )


# ==================================================
# 4. Reverse diffusion sampling
#
# pure Gaussian noise에서 시작해서
# x_99 → x_98 → ... → x_0
# ==================================================

@torch.no_grad()
def sample(
    n=5000,
):

    # ==============================================
    # x_T ~ N(0, 1)
    #
    # 실제 clean pose는 주지 않는다.
    # ==============================================

    x = torch.randn(
        n,
        1,
    )


    snapshots = {}


    # ==============================================
    # reverse process
    #
    # 99 → 98 → ... → 0
    # ==============================================

    for t_index in reversed(
        range(T)
    ):

        # 모든 generated sample이
        # 현재 같은 timestep에 있음

        t = torch.full(
            (n,),
            t_index,
            dtype=torch.long,
        )


        # ==========================================
        # 4-1. Predict current noise
        # ==========================================

        epsilon_hat = model(
            x,
            t,
        )


        alpha_t = alphas[
            t_index
        ]

        beta_t = betas[
            t_index
        ]

        a_bar_t = alpha_bar[
            t_index
        ]


        # ==========================================
        # 4-2. Reverse mean
        #
        # 현재 x_t에서
        # predicted noise를 제거해서
        #
        # x_(t-1)의 중심 위치를 계산
        # ==========================================

        mean = (

            1
            / torch.sqrt(
                alpha_t
            )

        ) * (

            x

            -

            beta_t
            / torch.sqrt(
                1 - a_bar_t
            )
            * epsilon_hat

        )


        # ==========================================
        # 4-3. stochastic reverse sampling
        #
        # 마지막 step 전까지는
        # 약간의 noise를 다시 추가
        # ==========================================

        if t_index > 0:

            a_bar_prev = alpha_bar[
                t_index - 1
            ]


            posterior_variance = (

                beta_t

                * (
                    1
                    - a_bar_prev
                )

                / (
                    1
                    - a_bar_t
                )
            )


            noise = torch.randn_like(
                x
            )


            x = (

                mean

                +

                torch.sqrt(
                    posterior_variance
                )
                * noise
            )


        else:

            # 마지막 t=0에서는
            # 더 이상 noise를 넣지 않음

            x = mean


        # ==========================================
        # snapshots
        # ==========================================

        if t_index in [
            99,
            75,
            50,
            25,
            0,
        ]:

            snapshots[
                t_index
            ] = x.clone()


    return (
        x.squeeze(1),
        snapshots,
    )


# ==================================================
# 5. Generate samples
# ==================================================

samples, snapshots = sample()


samples_np = (
    samples.numpy()
)


# ==================================================
# 6. Evaluate final distribution
# ==================================================

left_ratio = (
    samples_np < 0
).mean()


right_ratio = (
    samples_np >= 0
).mean()


distance_to_mode = np.minimum(

    np.abs(
        samples_np + 2
    ),

    np.abs(
        samples_np - 2
    ),
)


print(
    "\n=== GENERATED DISTRIBUTION ==="
)

print(
    "0° mode ratio:",
    left_ratio,
)

print(
    "180° mode ratio:",
    right_ratio,
)

print(
    "mean distance to nearest mode:",
    distance_to_mode.mean(),
)


# ==================================================
# 7. Decode scalar pose to angle
#
# 그냥 해석용
# ==================================================

angles = np.where(
    samples_np < 0,
    0,
    180,
)


print(
    "\nfirst 30 decoded poses:"
)

print(
    angles[:30]
)


# ==================================================
# 8. Inspect denoising process
# ==================================================

print(
    "\n=== DENOISING PROCESS ==="
)


for t_index in [
    99,
    75,
    50,
    25,
    0,
]:

    values = (
        snapshots[
            t_index
        ]
        .squeeze(1)
        .numpy()
    )


    left = (
        values < 0
    ).mean()


    error = np.minimum(

        np.abs(
            values + 2
        ),

        np.abs(
            values - 2
        ),

    ).mean()


    print(
        f"t={t_index:2d}",
        f"tau={t_index / (T - 1):.3f}",
        f"left={left:.3f}",
        f"mean_mode_error={error:.3f}",
    )