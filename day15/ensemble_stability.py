from collections import defaultdict

from autocrashfl_toy import (
    run_agent,
    ROOT_CAUSE,
)


# ============================================================
# Aggregate scores
#
# 한 run이 후보 집합 S를 반환하면
#
# 각 후보에 1 / |S|
#
# ============================================================

def aggregate_scores(
    runs,
):

    scores = defaultdict(float)

    for result in runs:

        S = result[
            "suspicious"
        ]

        if not S:
            continue

        weight = (
            1.0
            / len(S)
        )

        for file in S:

            scores[file] += (
                weight
            )

    return scores


# ============================================================
# Evaluate one ensemble
#
# top 후보가 여러 개 tie일 수도 있기 때문에
#
# 1. root cause가 top-score set 안에 있는가?
# 2. root cause가 unique top-1인가?
#
# 둘 다 측정한다.
# ============================================================

def evaluate_ensemble(
    R,
    ensemble_id,
    deep_search,
):

    runs = []

    for j in range(R):

        # 서로 다른 deterministic random seed
        run_id = (
            ensemble_id
            * 10000
            + j
        )

        result = run_agent(
            run_id=run_id,
            deep_search=deep_search,
        )

        runs.append(
            result
        )


    scores = aggregate_scores(
        runs
    )


    if not scores:

        return {
            "root_in_top": False,
            "root_unique_top": False,
            "confidence": 0.0,
        }


    max_score = max(
        scores.values()
    )


    top_set = {

        file

        for file, score
        in scores.items()

        if abs(
            score - max_score
        ) < 1e-12
    }


    root_in_top = (
        ROOT_CAUSE
        in top_set
    )


    root_unique_top = (
        top_set
        == {ROOT_CAUSE}
    )


    confidence = (
        max_score
        / R
    )


    return {
        "root_in_top":
        root_in_top,

        "root_unique_top":
        root_unique_top,

        "confidence":
        confidence,
    }


# ============================================================
# Repeat many independent ensembles
# ============================================================

def experiment(
    deep_search,
    ensemble_trials=1000,
):

    print(
        "\n================================"
    )

    print(
        "DEEP SEARCH:",
        deep_search
    )

    print(
        "================================"
    )


    R_VALUES = [
        1,
        2,
        3,
        5,
        10,
        20,
    ]


    for R in R_VALUES:

        root_in_top_count = 0

        unique_top_count = 0

        confidence_sum = 0.0


        for ensemble_id in range(
            ensemble_trials
        ):

            result = (
                evaluate_ensemble(
                    R=R,
                    ensemble_id=ensemble_id,
                    deep_search=deep_search,
                )
            )


            if result[
                "root_in_top"
            ]:

                root_in_top_count += 1


            if result[
                "root_unique_top"
            ]:

                unique_top_count += 1


            confidence_sum += (
                result[
                    "confidence"
                ]
            )


        root_in_top_rate = (
            root_in_top_count
            / ensemble_trials
        )


        unique_top_rate = (
            unique_top_count
            / ensemble_trials
        )


        avg_confidence = (
            confidence_sum
            / ensemble_trials
        )


        print(
            f"R={R:2d} "
            f"root_in_top="
            f"{root_in_top_rate:.3f} "
            f"unique_top1="
            f"{unique_top_rate:.3f} "
            f"avg_conf="
            f"{avg_confidence:.3f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    experiment(
        deep_search=True
    )

    experiment(
        deep_search=False
    )