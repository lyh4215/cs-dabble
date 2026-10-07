from dataclasses import dataclass


# ============================================================
# CROWN 2.0 Issue #27 — compressed reproduction
#
# IMPORTANT:
# 실제 CROWN source code가 아님.
#
# 논문의 핵심 failure mechanism만 압축:
#
#   wrong shift typing
#       ↓
#   wrong symbolic bit-width
#       ↓
#   downstream code에서 문제 발생
# ============================================================


@dataclass
class Expr:
    op: str

    lhs_bits: int
    rhs_bits: int


@dataclass
class Formula:
    op: str
    bits: int


# ============================================================
# Tests
#
# expected = symbolic result bit width
#
# 핵심 failing case:
#
#   1 << (unsigned long)b
#
#   lhs = 32
#   rhs = 64
#
# shift result는 lhs width인 32여야 한다고 가정.
# ============================================================

TESTS = [

    # --------------------------------------------------------
    # Failing cases in buggy model
    # --------------------------------------------------------

    (
        "shift_32_64",
        Expr(
            op="<<",
            lhs_bits=32,
            rhs_bits=64,
        ),
        32,
    ),

    (
        "shift_right_32_64",
        Expr(
            op=">>",
            lhs_bits=32,
            rhs_bits=64,
        ),
        32,
    ),

    # --------------------------------------------------------
    # Passing arithmetic cases
    #
    # toy에서는 usual arithmetic conversion을
    # max(width)로 단순화한다.
    # --------------------------------------------------------

    (
        "add_32_64",
        Expr(
            op="+",
            lhs_bits=32,
            rhs_bits=64,
        ),
        64,
    ),

    (
        "sub_32_64",
        Expr(
            op="-",
            lhs_bits=32,
            rhs_bits=64,
        ),
        64,
    ),

    # --------------------------------------------------------
    # Passing shifts
    # --------------------------------------------------------

    (
        "shift_32_32",
        Expr(
            op="<<",
            lhs_bits=32,
            rhs_bits=32,
        ),
        32,
    ),

    (
        "shift_64_32",
        Expr(
            op="<<",
            lhs_bits=64,
            rhs_bits=32,
        ),
        64,
    ),
]


# ============================================================
# Mutation modes
#
# BASE
#   buggy implementation
#
# FIX_ROOT
#   root-cause line을 제대로 수정
#
# FORCE_32_DOWNSTREAM
#   downstream에서 무조건 32-bit로 만들어
#   symptom을 억지로 고치는 mutation
#
# BROKEN_ARITH
#   unrelated/wrong mutation
# ============================================================

BASE = "base"

FIX_ROOT = "fix_root"

FORCE_32_DOWNSTREAM = (
    "force_32_downstream"
)

BROKEN_ARITH = (
    "broken_arith"
)


# ============================================================
# ROOT CAUSE
#
# 실제 논문의 핵심에 대응하는 부분.
#
# BUG:
#
# 모든 binary operator가
#
#   result width = max(lhs, rhs)
#
# 라고 잘못 가정.
#
# 그러면:
#
#   32 << 64
#
# 를
#
#   64 bit result
#
# 로 modeling한다.
# ============================================================

def infer_result_bits(
    expr,
    mode,
):

    # --------------------------------------------------------
    # Root-cause fixing mutation
    # --------------------------------------------------------

    if mode == FIX_ROOT:

        if expr.op in {
            "<<",
            ">>",
        }:

            # shift result follows lhs
            return expr.lhs_bits


    # --------------------------------------------------------
    # unrelated bad mutation
    # --------------------------------------------------------

    if (
        mode == BROKEN_ARITH
        and expr.op in {
            "+",
            "-",
        }
    ):

        return expr.lhs_bits


    # --------------------------------------------------------
    # BUGGY GENERAL RULE
    #
    # shift에도 이걸 적용해버리는 것이 문제.
    # --------------------------------------------------------

    return max(
        expr.lhs_bits,
        expr.rhs_bits,
    )


# ============================================================
# Symbolic modeling
# ============================================================

def model_binary(
    expr,
    mode,
):

    bits = infer_result_bits(
        expr,
        mode,
    )

    return Formula(
        op=expr.op,
        bits=bits,
    )


# ============================================================
# Downstream processing
#
# 실제 CROWN은 unary_expression.cc 쪽에서 crash가
# 나타났다고 논문이 설명한다.
#
# 여기서는 그 실제 crash implementation을 모르므로
# "downstream consumer"라는 toy abstraction으로 만든다.
#
# FORCE_32_DOWNSTREAM은:
#
#   원인을 고치는 게 아니라
#   downstream에서 무조건 32 bit로 강제하는
#   나쁜 symptom-level mutation이다.
# ============================================================

def downstream_process(
    formula,
    expected_bits,
    mode,
):

    if (
        mode
        == FORCE_32_DOWNSTREAM
    ):

        formula = Formula(
            op=formula.op,
            bits=32,
        )


    # --------------------------------------------------------
    # width invariant violation
    #
    # toy에서 crash manifestation 역할
    # --------------------------------------------------------

    if (
        formula.bits
        != expected_bits
    ):

        raise RuntimeError(
            "symbolic width mismatch"
        )


    return formula.bits


# ============================================================
# Full pipeline
#
# expression
#      ↓
# infer result width
#      ↓
# symbolic Formula
#      ↓
# downstream consumer
# ============================================================

def execute(
    expr,
    expected_bits,
    mode,
):

    try:

        formula = model_binary(
            expr,
            mode,
        )

        result = downstream_process(
            formula,
            expected_bits,
            mode,
        )

        return (
            result == expected_bits
        )

    except RuntimeError:

        return False


# ============================================================
# Original buggy program
# ============================================================

def run_all(mode):

    results = []

    for (
        name,
        expr,
        expected,
    ) in TESTS:

        passed = execute(
            expr,
            expected,
            mode,
        )

        results.append(
            passed
        )

    return results


BASE_RESULTS = run_all(
    BASE
)


print(
    "================================"
)

print(
    "BUGGY CROWN-LIKE MODEL"
)

print(
    "================================"
)


for (
    test,
    passed,
) in zip(
    TESTS,
    BASE_RESULTS,
):

    name, expr, expected = test

    predicted = infer_result_bits(
        expr,
        BASE,
    )

    print(
        f"{name:20s} "
        f"{expr.lhs_bits:2d} "
        f"{expr.op:2s} "
        f"{expr.rhs_bits:2d} "
        f"pred={predicted:2d} "
        f"expected={expected:2d} "
        f"{'PASS' if passed else 'FAIL'}"
    )


NUM_FAIL = sum(
    not x
    for x in BASE_RESULTS
)

NUM_PASS = sum(
    BASE_RESULTS
)


print()

print(
    "PASS:",
    NUM_PASS
)

print(
    "FAIL:",
    NUM_FAIL
)


# ============================================================
# Mutants
#
# 각 mutation을 source location에 대응시킨다.
#
# line 20:
#     result width inference
#
# line 40:
#     downstream manifestation
#
# line 24:
#     unrelated arithmetic rule
# ============================================================

MUTANTS = [

    {
        "name":
        "correct shift typing",

        "line":
        20,

        "mode":
        FIX_ROOT,
    },

    {
        "name":
        "force downstream to 32",

        "line":
        40,

        "mode":
        FORCE_32_DOWNSTREAM,
    },

    {
        "name":
        "change arithmetic typing",

        "line":
        24,

        "mode":
        BROKEN_ARITH,
    },
]


# ============================================================
# Mutation execution
# ============================================================

for mutant in MUTANTS:

    results = run_all(
        mutant["mode"]
    )

    mutant[
        "results"
    ] = results


    mutant[
        "f_to_p"
    ] = sum(

        (
            not original
            and changed
        )

        for (
            original,
            changed,
        )

        in zip(
            BASE_RESULTS,
            results,
        )
    )


    mutant[
        "p_to_f"
    ] = sum(

        (
            original
            and not changed
        )

        for (
            original,
            changed,
        )

        in zip(
            BASE_RESULTS,
            results,
        )
    )


# ============================================================
# Show mutant behavior
# ============================================================

print(
    "\n================================"
)

print(
    "MUTATION EFFECTS"
)

print(
    "================================"
)


for m in MUTANTS:

    print(
        f"\nline {m['line']}: "
        f"{m['name']}"
    )

    print(
        "results:",
        [
            "P"
            if x
            else "F"

            for x
            in m["results"]
        ],
    )

    print(
        "F -> P:",
        m["f_to_p"],
    )

    print(
        "P -> F:",
        m["p_to_f"],
    )


# ============================================================
# MUSE alpha
# ============================================================

total_f2p = sum(
    m["f_to_p"]
    for m in MUTANTS
)

total_p2f = sum(
    m["p_to_f"]
    for m in MUTANTS
)


alpha = (
    total_f2p
    * NUM_PASS
    /
    (
        total_p2f
        * NUM_FAIL
    )
)


print(
    "\n================================"
)

print(
    "GLOBAL STATISTICS"
)

print(
    "================================"
)

print(
    "F -> P:",
    total_f2p
)

print(
    "P -> F:",
    total_p2f
)

print(
    "alpha:",
    alpha
)


# ============================================================
# MUSE score
# ============================================================

def muse_score(m):

    return (

        m["f_to_p"]
        / NUM_FAIL

        -

        alpha
        * m["p_to_f"]
        / NUM_PASS
    )


ranking = []

for m in MUTANTS:

    ranking.append(
        (
            muse_score(m),
            m,
        )
    )


ranking.sort(
    key=lambda x: x[0],
    reverse=True,
)


print(
    "\n================================"
)

print(
    "FAULT LOCALIZATION"
)

print(
    "================================"
)


for rank, (
    score,
    mutant,
) in enumerate(
    ranking,
    1,
):

    marker = (

        "  <-- ROOT CAUSE"

        if mutant["line"] == 20

        else ""
    )

    print(
        f"{rank}. "
        f"line {mutant['line']:2d} "
        f"score={score:+.3f} "
        f"{mutant['name']}"
        f"{marker}"
    )


# ============================================================
# Error propagation visualization
# ============================================================

print(
    "\n================================"
)

print(
    "ISSUE #27 CONCEPTUAL FLOW"
)

print(
    "================================"
)


bad = Expr(
    op="<<",
    lhs_bits=32,
    rhs_bits=64,
)


bad_width = infer_result_bits(
    bad,
    BASE,
)


fixed_width = infer_result_bits(
    bad,
    FIX_ROOT,
)


print(
    "input:"
)

print(
    "  1 << (unsigned long)b"
)

print(
    "  lhs bits = 32"
)

print(
    "  rhs bits = 64"
)


print(
    "\nBUGGY modeling:"
)

print(
    "  max(32, 64)"
    f" -> {bad_width}"
)

print(
    "  wrong symbolic Formula"
)

print(
    "  ↓"
)

print(
    "  downstream width mismatch"
)

print(
    "  ↓"
)

print(
    "  crash / failure manifestation"
)


print(
    "\nCORRECT modeling:"
)

print(
    "  shift result follows lhs"
)

print(
    f"  -> {fixed_width} bits"
)

print(
    "  ↓"
)

print(
    "  downstream invariant satisfied"
)