import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

SEED = 42

DT = 0.01          # IMU: 100 Hz
TOTAL_TIME = 60.0

WIFI_INTERVAL = 1.0    # Wi-Fi 위치 관측: 1 Hz

TRUE_ACC_BIAS = 0.03   # m/s^2
ACC_NOISE_STD = 0.08   # m/s^2

WIFI_NOISE_STD = 1.5   # meters
OUTLIER_PROB = 0.12
OUTLIER_STD = 15.0

rng = np.random.default_rng(SEED)


# ============================================================
# 1. Ground-truth trajectory
# ============================================================

t = np.arange(
    0,
    TOTAL_TIME,
    DT
)

n = len(t)


true_acc = np.zeros(n)


# 여러 구간에서 가속 / 감속
true_acc[
    (t >= 2) & (t < 6)
] = 0.7

true_acc[
    (t >= 12) & (t < 16)
] = -0.7

true_acc[
    (t >= 25) & (t < 29)
] = 0.5

true_acc[
    (t >= 38) & (t < 42)
] = -0.5


true_vel = np.zeros(n)
true_pos = np.zeros(n)


for i in range(1, n):

    true_vel[i] = (
        true_vel[i - 1]
        + true_acc[i - 1] * DT
    )

    true_pos[i] = (
        true_pos[i - 1]
        + true_vel[i - 1] * DT
        + 0.5
        * true_acc[i - 1]
        * DT**2
    )


# ============================================================
# 2. Accelerometer measurement
#
# measured acceleration =
# true acceleration
# + constant bias
# + random noise
# ============================================================

acc_measurement = (
    true_acc
    + TRUE_ACC_BIAS
    + rng.normal(
        0,
        ACC_NOISE_STD,
        n
    )
)


# ============================================================
# 3. IMU-only dead reckoning
#
# bias를 모른다고 가정하고
# 측정된 acceleration을 그대로 두 번 적분
# ============================================================

imu_vel = np.zeros(n)
imu_pos = np.zeros(n)


for i in range(1, n):

    imu_vel[i] = (
        imu_vel[i - 1]
        + acc_measurement[i - 1]
        * DT
    )

    imu_pos[i] = (
        imu_pos[i - 1]
        + imu_vel[i - 1]
        * DT
        + 0.5
        * acc_measurement[i - 1]
        * DT**2
    )


# ============================================================
# 4. Wi-Fi absolute position measurements
# ============================================================

wifi_step = int(
    WIFI_INTERVAL / DT
)

wifi_indices = np.arange(
    0,
    n,
    wifi_step
)


wifi_measurements = (
    true_pos[wifi_indices]
    + rng.normal(
        0,
        WIFI_NOISE_STD,
        len(wifi_indices)
    )
)


# 일부 Wi-Fi 관측을 크게 망가뜨림
is_outlier = (
    rng.random(
        len(wifi_indices)
    )
    < OUTLIER_PROB
)

wifi_measurements[
    is_outlier
] += rng.normal(
    0,
    OUTLIER_STD,
    is_outlier.sum()
)


# ============================================================
# 5. Kalman filter
#
# state:
#
#     [ position ]
# x = [ velocity ]
#     [ acc bias ]
#
#
# 실제 acceleration:
#
# a_true ≈ a_measured - bias
#
# 따라서:
#
# p' = p + v dt
#      + 1/2 (a_measured - bias) dt^2
#
# v' = v
#      + (a_measured - bias) dt
#
# bias' = bias
# ============================================================

F = np.array([
    [
        1.0,
        DT,
        -0.5 * DT**2
    ],
    [
        0.0,
        1.0,
        -DT
    ],
    [
        0.0,
        0.0,
        1.0
    ]
])


B = np.array([
    0.5 * DT**2,
    DT,
    0.0
])


H = np.array([
    [1.0, 0.0, 0.0]
])


# process noise
Q = np.diag([
    1e-5,
    1e-4,
    1e-7
])


# Wi-Fi measurement variance
R = np.array([
    [WIFI_NOISE_STD**2]
])


def run_kalman(
    reject_outliers=False
):

    x = np.array([
        0.0,    # position
        0.0,    # velocity
        0.0     # bias estimate
    ])


    P = np.diag([
        1.0,
        1.0,
        0.1
    ])


    positions = np.zeros(n)
    velocities = np.zeros(n)
    biases = np.zeros(n)


    wifi_ptr = 0

    accepted = 0
    rejected = 0


    for i in range(n):

        # --------------------------------------------
        # Prediction using IMU
        # --------------------------------------------

        if i > 0:

            x = (
                F @ x
                + B * acc_measurement[i - 1]
            )

            P = (
                F @ P @ F.T
                + Q
            )


        # --------------------------------------------
        # Wi-Fi measurement available?
        # --------------------------------------------

        if (
            wifi_ptr < len(wifi_indices)
            and i == wifi_indices[wifi_ptr]
        ):

            z = np.array([
                wifi_measurements[wifi_ptr]
            ])


            # innovation
            y = (
                z
                - H @ x
            )


            # innovation covariance
            S = (
                H @ P @ H.T
                + R
            )


            accept = True


            # ----------------------------------------
            # Innovation gating
            #
            # |innovation| >
            # 3 * expected std
            #
            # 이면 outlier라고 판단
            # ----------------------------------------

            if reject_outliers:

                innovation_std = np.sqrt(
                    S[0, 0]
                )

                if (
                    abs(y[0])
                    > 3.0 * innovation_std
                ):

                    accept = False


            if accept:

                K = (
                    P
                    @ H.T
                    @ np.linalg.inv(S)
                )

                x = (
                    x
                    + (K @ y)
                )

                P = (
                    (
                        np.eye(3)
                        - K @ H
                    )
                    @ P
                )

                accepted += 1

            else:

                rejected += 1


            wifi_ptr += 1


        positions[i] = x[0]
        velocities[i] = x[1]
        biases[i] = x[2]


    return (
        positions,
        velocities,
        biases,
        accepted,
        rejected
    )


# ============================================================
# 6. Run filters
# ============================================================

(
    kf_pos,
    kf_vel,
    kf_bias,
    kf_accepted,
    kf_rejected
) = run_kalman(
    reject_outliers=False
)


(
    robust_pos,
    robust_vel,
    robust_bias,
    robust_accepted,
    robust_rejected
) = run_kalman(
    reject_outliers=True
)


# ============================================================
# 7. Metrics
# ============================================================

def rmse(pred, truth):

    return np.sqrt(
        np.mean(
            (pred - truth) ** 2
        )
    )


imu_rmse = rmse(
    imu_pos,
    true_pos
)

kf_rmse = rmse(
    kf_pos,
    true_pos
)

robust_rmse = rmse(
    robust_pos,
    true_pos
)


print(
    "=== SENSOR FUSION RESULT ==="
)

print()

print(
    f"True accelerometer bias : "
    f"{TRUE_ACC_BIAS:.4f} m/s^2"
)

print()

print(
    f"Wi-Fi observations      : "
    f"{len(wifi_indices)}"
)

print(
    f"Injected outliers       : "
    f"{is_outlier.sum()}"
)

print()

print(
    "Position RMSE"
)

print(
    f"  IMU only              : "
    f"{imu_rmse:.3f} m"
)

print(
    f"  Kalman + Wi-Fi        : "
    f"{kf_rmse:.3f} m"
)

print(
    f"  Robust Kalman         : "
    f"{robust_rmse:.3f} m"
)

print()

print(
    "Robust filter"
)

print(
    f"  accepted Wi-Fi        : "
    f"{robust_accepted}"
)

print(
    f"  rejected Wi-Fi        : "
    f"{robust_rejected}"
)

print()

print(
    f"Final bias estimate     : "
    f"{robust_bias[-1]:.4f} m/s^2"
)


# ============================================================
# 8. Plot
# ============================================================

os.makedirs(
    "dist",
    exist_ok=True
)


# ------------------------------------------------------------
# Position
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    t,
    true_pos,
    label="Ground truth"
)

plt.plot(
    t,
    imu_pos,
    label="IMU only"
)

plt.plot(
    t,
    kf_pos,
    label="Kalman + Wi-Fi"
)

plt.plot(
    t,
    robust_pos,
    label="Robust Kalman"
)

plt.scatter(
    t[wifi_indices],
    wifi_measurements,
    s=14,
    label="Wi-Fi measurements"
)

plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title(
    "1D Indoor Localization"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "dist/position.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# Position error
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    t,
    imu_pos - true_pos,
    label="IMU only"
)

plt.plot(
    t,
    kf_pos - true_pos,
    label="Kalman + Wi-Fi"
)

plt.plot(
    t,
    robust_pos - true_pos,
    label="Robust Kalman"
)

plt.axhline(
    0
)

plt.xlabel("Time (s)")
plt.ylabel("Position error (m)")
plt.title(
    "Localization Error"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "dist/error.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# Bias estimation
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    t,
    robust_bias,
    label="Estimated bias"
)

plt.axhline(
    TRUE_ACC_BIAS,
    linestyle="--",
    label="True bias"
)

plt.xlabel("Time (s)")
plt.ylabel(
    "Accelerometer bias (m/s²)"
)

plt.title(
    "Online Accelerometer Bias Estimation"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "dist/bias.png",
    dpi=160
)

plt.close()


print()
print(
    "Saved:"
)

print(
    "  dist/position.png"
)

print(
    "  dist/error.png"
)

print(
    "  dist/bias.png"
)