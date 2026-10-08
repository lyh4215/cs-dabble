from dataclasses import dataclass
import textwrap


# ============================================================
# 1. BUGGY PROGRAM
#
# BUG:
#
# interior value는 그대로 x를 반환해야 하는데
#
#     return x + 1
#
# 이 되어 있음.
# ============================================================

SOURCE = """def clamp(x, lo, hi):
    if x < lo:
        return lo

    if x > hi:
        return hi

    return x + 1
"""


# ============================================================
# 2. TESTS
#
# bug-revealing test는 하나.
#
# regression tests는 기존 정상 행동을 보존하는지 검사.
# ============================================================

FAILING_TEST = (
    (5, 0, 10),
    5,
)


REGRESSION_TESTS = [

    (
        (-3, 0, 10),
        0,
    ),

    (
        (12, 0, 10),
        10,
    ),
]


# ============================================================
# 3. Compile source
# ============================================================

def compile_program(source):

    namespace = {}

    exec(
        source,
        namespace,
    )

    return namespace["clamp"]


# ============================================================
# 4. Run one test
# ============================================================

def run_test(fn, test):

    args, expected = test

    try:

        actual = fn(*args)

        return {
            "args": args,
            "expected": expected,
            "actual": actual,
            "passed": (
                actual == expected
            ),
        }

    except Exception as e:

        return {
            "args": args,
            "expected": expected,
            "actual": repr(e),
            "passed": False,
        }


# ============================================================
# 5. Run bug-revealing + regression suite
# ============================================================

def run_suite(source):

    fn = compile_program(
        source
    )

    failing = run_test(
        fn,
        FAILING_TEST,
    )

    regressions = [

        run_test(
            fn,
            test,
        )

        for test
        in REGRESSION_TESTS
    ]

    return {
        "failing_test": failing,
        "regressions": regressions,
    }


# ============================================================
# 6. Simple REPLACE DSL
#
# AutoSD 논문의 실제 DSL에는
#
# REPLACE
# ADD
# DEL
# RUN
#
# 등이 있음.
#
# 여기서는 첫 실험이므로 REPLACE + RUN만 구현.
# ============================================================

def replace(
    source,
    old,
    new,
):

    if old not in source:

        raise ValueError(
            f"cannot find: {old}"
        )

    return source.replace(
        old,
        new,
        1,
    )


# ============================================================
# 7. Hypothesis
#
# Scientific Debugging:
#
# Hypothesis
# Prediction
# Experiment
# Observation
# Conclusion
# ============================================================

@dataclass
class Hypothesis:

    name: str

    hypothesis: str

    prediction: str

    old: str

    new: str


HYPOTHESES = [

    Hypothesis(

        name="H1",

        hypothesis=(
            "The upper-bound condition is "
            "off by one."
        ),

        prediction=(
            "Changing `x > hi` to "
            "`x >= hi` should make the "
            "bug-revealing test pass."
        ),

        old="if x > hi:",

        new="if x >= hi:",
    ),


    Hypothesis(

        name="H2",

        hypothesis=(
            "The lower-bound condition is "
            "off by one."
        ),

        prediction=(
            "Changing `x < lo` to "
            "`x <= lo` should make the "
            "bug-revealing test pass."
        ),

        old="if x < lo:",

        new="if x <= lo:",
    ),


    Hypothesis(

        name="H3",

        hypothesis=(
            "The interior return value has "
            "an off-by-one error."
        ),

        prediction=(
            "Changing `return x + 1` to "
            "`return x` should make the "
            "bug-revealing test pass "
            "without breaking regression tests."
        ),

        old="return x + 1",

        new="return x",
    ),
]


# ============================================================
# 8. Decide conclusion
#
# 단순히 failing test만 PASS하면 안 됨.
#
# regression tests까지 모두 PASS해야
# hypothesis를 강하게 support한다고 가정.
# ============================================================

def conclude(observation):

    failing_passed = (
        observation[
            "failing_test"
        ]["passed"]
    )

    regression_passed = all(

        result["passed"]

        for result
        in observation[
            "regressions"
        ]
    )


    if (
        failing_passed
        and regression_passed
    ):

        return "SUPPORTED"


    if not failing_passed:

        return "REJECTED"


    return "UNDECIDED"


# ============================================================
# 9. Scientific Debugging Loop
# ============================================================

def debug():

    print(
        "================================"
    )

    print(
        "ORIGINAL PROGRAM"
    )

    print(
        "================================"
    )

    print(
        SOURCE
    )


    original = run_suite(
        SOURCE
    )


    print(
        "================================"
    )

    print(
        "INITIAL FAILURE"
    )

    print(
        "================================"
    )

    print(
        original[
            "failing_test"
        ]
    )


    print(
        "\nRegression tests:"
    )

    for r in original[
        "regressions"
    ]:

        print(
            r
        )


    # --------------------------------------------------------
    # Hypothesize -> Experiment -> Observe -> Conclude
    # --------------------------------------------------------

    for h in HYPOTHESES:

        print(
            "\n\n================================"
        )

        print(
            h.name
        )

        print(
            "================================"
        )


        # ----------------------------------------------------
        # HYPOTHESIS
        # ----------------------------------------------------

        print(
            "\n[HYPOTHESIS]"
        )

        print(
            h.hypothesis
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        print(
            "\n[PREDICTION]"
        )

        print(
            h.prediction
        )


        # ----------------------------------------------------
        # EXPERIMENT
        #
        # REPLACE + RUN
        # ----------------------------------------------------

        print(
            "\n[EXPERIMENT]"
        )

        print(
            f'REPLACE("{h.old}", "{h.new}")'
        )

        print(
            "RUN"
        )


        mutant = replace(
            SOURCE,
            h.old,
            h.new,
        )


        # ----------------------------------------------------
        # OBSERVATION
        #
        # 실제 Python 실행 결과
        # ----------------------------------------------------

        observation = run_suite(
            mutant
        )


        print(
            "\n[OBSERVATION]"
        )


        failing = observation[
            "failing_test"
        ]


        print(
            "bug-revealing test:"
        )

        print(
            f"  args={failing['args']}"
        )

        print(
            f"  expected={failing['expected']}"
        )

        print(
            f"  actual={failing['actual']}"
        )

        print(
            f"  result="
            f"{'PASS' if failing['passed'] else 'FAIL'}"
        )


        print(
            "\nregression tests:"
        )


        for r in observation[
            "regressions"
        ]:

            print(
                f"  {r['args']} -> "
                f"{'PASS' if r['passed'] else 'FAIL'}"
            )


        # ----------------------------------------------------
        # CONCLUSION
        # ----------------------------------------------------

        conclusion = conclude(
            observation
        )


        print(
            "\n[CONCLUSION]"
        )

        print(
            conclusion
        )


        # ----------------------------------------------------
        # DEBUGGING DONE
        # ----------------------------------------------------

        if conclusion == "SUPPORTED":

            print(
                "\n<DEBUGGING DONE>"
            )


            print(
                "\n================================"
            )

            print(
                "PATCH"
            )

            print(
                "================================"
            )

            print(
                f"- {h.old}"
            )

            print(
                f"+ {h.new}"
            )


            print(
                "\n================================"
            )

            print(
                "EXPLANATION TRACE"
            )

            print(
                "================================"
            )


            print(
                "Hypothesis:"
            )

            print(
                h.hypothesis
            )


            print(
                "\nPrediction:"
            )

            print(
                h.prediction
            )


            print(
                "\nObservation:"
            )

            print(
                "The failing test now passes "
                "and all regression tests remain "
                "passing."
            )


            print(
                "\nConclusion:"
            )

            print(
                "The hypothesis is supported."
            )

            return


    print(
        "\nDEBUGGING FAILED"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    debug()