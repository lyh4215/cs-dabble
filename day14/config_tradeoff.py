rows = [
    # line_ratio, mutants_per_line, top1, top3, top5, mfr
    (1.00, 10, 43.8, 67.0, 77.7, 6.56),

    (0.70, 10, 43.1, 66.3, 76.3, 6.78),
    (0.70,  9, 42.5, 66.4, 76.8, 6.54),
    (0.70,  8, 42.5, 65.9, 76.3, 6.46),
    (0.70,  7, 42.3, 66.2, 76.4, 6.69),
    (0.70,  6, 42.6, 66.2, 76.7, 6.67),
    (0.70,  5, 42.2, 65.9, 76.7, 6.75),
    (0.70,  4, 42.5, 65.9, 76.5, 6.79),
    (0.70,  3, 42.4, 66.7, 76.2, 6.93),
    (0.70,  2, 42.3, 65.6, 76.4, 6.90),
    (0.70,  1, 41.8, 64.2, 76.4, 6.68),
]


baseline = rows[0]

baseline_proxy = (
    baseline[0]
    * baseline[1]
)

baseline_top1 = baseline[2]


print(
    "============================================================"
)

print(
    "CONFIGURATION TRADE-OFF"
)

print(
    "============================================================"
)


results = []

for (
    line_ratio,
    mutants,
    top1,
    top3,
    top5,
    mfr,
) in rows:

    # --------------------------------------------------------
    # 아주 단순한 mutant-count cost proxy
    #
    # 전체 mutant 개수는 대략:
    #
    # selected lines
    #     ×
    # mutants per line
    #
    # 에 비례한다고 생각.
    #
    # 실제 CPU cost가 정확히 이 비율이라는 뜻은 아님.
    # --------------------------------------------------------

    cost_proxy = (
        line_ratio
        * mutants
    )

    normalized_cost = (
        cost_proxy
        / baseline_proxy
    )

    top1_loss = (
        baseline_top1
        - top1
    )

    results.append(
        {
            "line_ratio": line_ratio,
            "mutants": mutants,
            "top1": top1,
            "cost": normalized_cost,
            "loss": top1_loss,
        }
    )

    print(
        f"lines={line_ratio:4.0%} "
        f"mutants/line={mutants:2d} "
        f"cost_proxy={normalized_cost:6.1%} "
        f"Top1={top1:4.1f}% "
        f"loss={top1_loss:+4.1f}pp"
    )


# ============================================================
# Suppose we tolerate at most 1.5 percentage-point Top-1 loss.
#
# 이건 우리가 분석을 위해 정한 threshold이고
# 논문의 criterion은 아님.
# ============================================================

MAX_LOSS = 1.5


acceptable = [
    r
    for r in results
    if r["loss"] <= MAX_LOSS
]


best = min(
    acceptable,
    key=lambda r: r["cost"]
)


print(
    "\n============================================================"
)

print(
    "CHEAPEST CONFIG WITH <= 1.5pp TOP-1 LOSS"
)

print(
    "============================================================"
)

print(
    "line ratio:",
    f'{best["line_ratio"]:.0%}'
)

print(
    "mutants per line:",
    best["mutants"]
)

print(
    "mutant-count cost proxy:",
    f'{best["cost"]:.1%}'
)

print(
    "Top-1:",
    f'{best["top1"]:.1f}%'
)

print(
    "Top-1 loss:",
    f'{best["loss"]:.1f}pp'
)