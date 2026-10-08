from z3 import (
    Int,
    IntVal,
    If,
    Optimize,
    Solver,
    sat,
)


# ============================================================
# Specification
#
# 올바른 clamp:
#
# x < 0   -> 0
# x > 10  -> 10
# otherwise -> x
# ============================================================

LO = 0
HI = 10


def spec_expr(x):

    return If(
        x < LO,
        LO,

        If(
            x > HI,
            HI,
            x,
        ),
    )


# ============================================================
# Repair template
#
# 원래 buggy code:
#
#     return x + 1
#
# 를
#
#     return a*x + b
#
# 형태에서 합성한다고 하자.
#
#
# 전체 candidate:
#
# if x < 0:
#     return 0
#
# if x > 10:
#     return 10
#
# return a*x + b
# ============================================================

def candidate_expr(
    x,
    a,
    b,
):

    return If(
        x < LO,
        LO,

        If(
            x > HI,
            HI,

            a * x + b,
        ),
    )


# ============================================================
# Concrete versions
# ============================================================

def spec(x):

    if x < LO:
        return LO

    if x > HI:
        return HI

    return x


def candidate(
    x,
    a,
    b,
):

    if x < LO:
        return LO

    if x > HI:
        return HI

    return (
        a * x + b
    )


# ============================================================
# Initial examples
#
# 개발자가 가진 test suite라고 생각.
#
# 중요:
#
# (-3, 0), (12, 10)은
# outer branch에서 끝나므로
#
# a, b를 거의 constrain하지 않는다.
#
# 실제 interior repair를 constrain하는 건
# 처음에는 x=5 하나뿐.
# ============================================================

INITIAL_EXAMPLES = [

    (
        -3,
        0,
    ),

    (
        12,
        10,
    ),

    (
        5,
        5,
    ),
]


# ============================================================
# SYNTHESIZER
#
# 현재 examples를 모두 만족하는
#
# a, b
#
# 를 찾는다.
#
#
# 여러 solution이 있을 수 있기 때문에
# 단순한 candidate를 우선하도록
#
# |a| 최소화
# |b| 최소화
#
# 한다.
#
# 그래서 첫 iteration에서는
#
#     a = 0
#     b = 5
#
# 즉
#
#     return 5
#
# 같은 overfitting patch가 나올 가능성이 높다.
# ============================================================

def synthesize(examples):

    a = Int("a")
    b = Int("b")

    opt = Optimize()


    # --------------------------------------------------------
    # 모든 현재 example을 만족해야 함
    # --------------------------------------------------------

    for x_value, expected in examples:

        x = IntVal(
            x_value
        )

        opt.add(

            candidate_expr(
                x,
                a,
                b,
            )

            ==

            IntVal(
                expected
            )
        )


    # --------------------------------------------------------
    # simpler candidate preference
    #
    # |a| 최소화
    # --------------------------------------------------------

    abs_a = If(
        a >= 0,
        a,
        -a,
    )

    abs_b = If(
        b >= 0,
        b,
        -b,
    )

    opt.minimize(
        abs_a
    )

    opt.minimize(
        abs_b
    )


    result = opt.check()

    if result != sat:

        return None


    model = opt.model()


    a_value = model.eval(
        a,
        model_completion=True,
    ).as_long()


    b_value = model.eval(
        b,
        model_completion=True,
    ).as_long()


    return (
        a_value,
        b_value,
    )


# ============================================================
# VERIFIER
#
# candidate와 specification이 다른 입력이
# 존재하는지 묻는다.
#
#
# ∃x.
#
# Candidate(x) != Spec(x)
#
#
# SAT:
#     counterexample 존재
#
# UNSAT:
#     이 모델에서는 모든 integer x에 대해
#     equivalent
# ============================================================

def verify(
    a,
    b,
):

    x = Int(
        "counterexample_x"
    )

    solver = Solver()


    solver.add(

        candidate_expr(
            x,
            IntVal(a),
            IntVal(b),
        )

        !=

        spec_expr(x)
    )


    result = solver.check()


    if result == sat:

        model = solver.model()

        x_value = model.eval(
            x,
            model_completion=True,
        ).as_long()


        return {

            "x":
            x_value,

            "candidate":
            candidate(
                x_value,
                a,
                b,
            ),

            "expected":
            spec(
                x_value
            ),
        }


    return None


# ============================================================
# Pretty-print candidate
# ============================================================

def patch_string(
    a,
    b,
):

    if (
        a == 0
        and b == 0
    ):

        return "return 0"


    if a == 0:

        return (
            f"return {b}"
        )


    if (
        a == 1
        and b == 0
    ):

        return "return x"


    if (
        a == -1
        and b == 0
    ):

        return "return -x"


    if b == 0:

        return (
            f"return {a} * x"
        )


    if b > 0:

        return (
            f"return {a} * x + {b}"
        )


    return (
        f"return {a} * x - {-b}"
    )


# ============================================================
# CEGIS LOOP
# ============================================================

def cegis():

    examples = list(
        INITIAL_EXAMPLES
    )


    print(
        "================================"
    )

    print(
        "INITIAL EXAMPLES"
    )

    print(
        "================================"
    )


    for x, y in examples:

        print(
            f"x={x:3d} -> {y}"
        )


    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for iteration in range(
        1,
        11,
    ):

        print(
            "\n\n================================"
        )

        print(
            f"ITERATION {iteration}"
        )

        print(
            "================================"
        )


        # ====================================================
        # SYNTHESIS
        # ====================================================

        print(
            "\n[SYNTHESIS]"
        )


        result = synthesize(
            examples
        )


        if result is None:

            print(
                "No candidate exists."
            )

            return


        a, b = result


        print(
            f"a = {a}"
        )

        print(
            f"b = {b}"
        )


        print(
            "candidate patch:"
        )

        print(
            "   ",
            patch_string(
                a,
                b,
            )
        )


        # ====================================================
        # VERIFICATION
        # ====================================================

        print(
            "\n[VERIFICATION]"
        )


        print(
            "checking:"
        )

        print(
            "∃ x . Candidate(x) != Spec(x)"
        )


        counterexample = verify(
            a,
            b,
        )


        # ====================================================
        # VERIFIED
        # ====================================================

        if counterexample is None:

            print(
                "UNSAT"
            )

            print(
                "No counterexample exists."
            )


            print(
                "\n================================"
            )

            print(
                "CEGIS COMPLETE"
            )

            print(
                "================================"
            )


            print(
                "final patch:"
            )

            print(
                "   ",
                patch_string(
                    a,
                    b,
                )
            )


            print(
                "\nparameters:"
            )

            print(
                f"a={a}, b={b}"
            )


            print(
                "\nThe candidate is equivalent "
                "to the specification under "
                "this model."
            )


            return


        # ====================================================
        # COUNTEREXAMPLE
        # ====================================================

        print(
            "SAT"
        )

        print(
            "counterexample found:"
        )

        print(
            f"x = "
            f"{counterexample['x']}"
        )

        print(
            "candidate =",
            counterexample[
                "candidate"
            ],
        )

        print(
            "expected  =",
            counterexample[
                "expected"
            ],
        )


        # ====================================================
        # REFINE
        #
        # counterexample을 새로운 synthesis example로 추가
        # ====================================================

        new_example = (

            counterexample[
                "x"
            ],

            counterexample[
                "expected"
            ],
        )


        if (
            new_example
            not in examples
        ):

            examples.append(
                new_example
            )


        print(
            "\n[REFINEMENT]"
        )

        print(
            "add counterexample "
            "to examples:"
        )

        print(
            new_example
        )


        print(
            "\ncurrent examples:"
        )


        for x, y in examples:

            print(
                f"  {x:3d} -> {y}"
            )


    print(
        "Maximum iterations reached."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    cegis()