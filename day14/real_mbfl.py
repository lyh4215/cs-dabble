import ast
import copy
import math
import textwrap
from collections import defaultdict


# ============================================================
# 1. Buggy program
# ============================================================

SOURCE = """
def abs_diff(a, b):
    d = a - b
    if d < 0:
        d = d
    result = d
    return result
"""


# source line mapping
#
# line 3: d = a - b
# line 4: if d < 0
# line 5: d = d        <- BUG
# line 6: result = d
# line 7: return result


BUG_LINE = 5


# ============================================================
# 2. Tests
#
# original program:
#
# PASS: (5,3), (7,7), (10,2)
# FAIL: (3,5), (1,4)
# ============================================================

TESTS = [
    ((5, 3), 2),
    ((3, 5), 2),
    ((7, 7), 0),
    ((1, 4), 3),
    ((10, 2), 8),
]


# ============================================================
# 3. Compile AST -> Python function
# ============================================================

def compile_function(tree):

    tree = ast.fix_missing_locations(tree)

    code = compile(
        tree,
        filename="<mutant>",
        mode="exec",
    )

    namespace = {}

    exec(
        code,
        namespace,
    )

    return namespace["abs_diff"]


# ============================================================
# 4. Run all tests
# ============================================================

def run_tests(fn):

    results = []

    for args, expected in TESTS:

        try:
            actual = fn(*args)

            passed = (
                actual == expected
            )

        except Exception:
            passed = False

        results.append(passed)

    return results


# ============================================================
# 5. Original execution
# ============================================================

BASE_TREE = ast.parse(
    textwrap.dedent(SOURCE)
)

BASE_FN = compile_function(
    copy.deepcopy(BASE_TREE)
)

BASE_RESULTS = run_tests(
    BASE_FN
)


print(
    "================================"
)

print(
    "ORIGINAL TEST RESULTS"
)

print(
    "================================"
)


for i, (
    test,
    passed,
) in enumerate(
    zip(TESTS, BASE_RESULTS),
    1,
):

    args, expected = test

    print(
        f"T{i}: "
        f"args={args} "
        f"expected={expected} "
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
# 6. Mutation-site discovery
#
# 아주 작은 mutation operator set:
#
# BinOp:
#   a - b  <->  a + b
#
# Compare:
#   <  <->  >=
#
# Assignment:
#   x = y  ->  x = -y
#
# Return:
#   return x -> return -x
#
# 실제 mutation testing tool은 훨씬 많은 operator를 사용함.
# ============================================================

class MutationSiteFinder(
    ast.NodeVisitor
):

    def __init__(self):

        self.sites = []


    def visit_BinOp(self, node):

        if isinstance(
            node.op,
            (
                ast.Sub,
                ast.Add,
            ),
        ):

            self.sites.append(
                (
                    "binop",
                    node.lineno,
                )
            )

        self.generic_visit(node)


    def visit_Compare(self, node):

        if (
            len(node.ops) == 1
            and isinstance(
                node.ops[0],
                (
                    ast.Lt,
                    ast.GtE,
                ),
            )
        ):

            self.sites.append(
                (
                    "compare",
                    node.lineno,
                )
            )

        self.generic_visit(node)


    def visit_Assign(self, node):

        # x = y 형태만 mutation
        if isinstance(
            node.value,
            ast.Name,
        ):

            self.sites.append(
                (
                    "neg_assign",
                    node.lineno,
                )
            )

        self.generic_visit(node)


    def visit_Return(self, node):

        if isinstance(
            node.value,
            ast.Name,
        ):

            self.sites.append(
                (
                    "neg_return",
                    node.lineno,
                )
            )

        self.generic_visit(node)


finder = MutationSiteFinder()

finder.visit(
    BASE_TREE
)

MUTATION_SITES = (
    finder.sites
)


print(
    "\n================================"
)

print(
    "MUTATION SITES"
)

print(
    "================================"
)

for site in MUTATION_SITES:

    print(site)


# ============================================================
# 7. AST Mutator
# ============================================================

class Mutator(
    ast.NodeTransformer
):

    def __init__(
        self,
        kind,
        lineno,
    ):

        self.kind = kind
        self.lineno = lineno


    def visit_BinOp(self, node):

        self.generic_visit(node)

        if (
            self.kind == "binop"
            and node.lineno
            == self.lineno
        ):

            if isinstance(
                node.op,
                ast.Sub,
            ):

                node.op = ast.Add()

            elif isinstance(
                node.op,
                ast.Add,
            ):

                node.op = ast.Sub()

        return node


    def visit_Compare(self, node):

        self.generic_visit(node)

        if (
            self.kind == "compare"
            and node.lineno
            == self.lineno
        ):

            if isinstance(
                node.ops[0],
                ast.Lt,
            ):

                node.ops[0] = (
                    ast.GtE()
                )

            elif isinstance(
                node.ops[0],
                ast.GtE,
            ):

                node.ops[0] = (
                    ast.Lt()
                )

        return node


    def visit_Assign(self, node):

        self.generic_visit(node)

        if (
            self.kind
            == "neg_assign"
            and node.lineno
            == self.lineno
            and isinstance(
                node.value,
                ast.Name,
            )
        ):

            node.value = (
                ast.UnaryOp(
                    op=ast.USub(),
                    operand=node.value,
                )
            )

        return node


    def visit_Return(self, node):

        self.generic_visit(node)

        if (
            self.kind
            == "neg_return"
            and node.lineno
            == self.lineno
            and isinstance(
                node.value,
                ast.Name,
            )
        ):

            node.value = (
                ast.UnaryOp(
                    op=ast.USub(),
                    operand=node.value,
                )
            )

        return node


# ============================================================
# 8. Generate mutants + execute tests
# ============================================================

mutant_results = []


for mutant_id, (
    kind,
    lineno,
) in enumerate(
    MUTATION_SITES,
    1,
):

    tree = copy.deepcopy(
        BASE_TREE
    )

    mutator = Mutator(
        kind,
        lineno,
    )

    mutant_tree = (
        mutator.visit(tree)
    )

    mutant_fn = (
        compile_function(
            mutant_tree
        )
    )

    results = run_tests(
        mutant_fn
    )


    # --------------------------------------------------------
    # Compare original test outcome vs mutant
    #
    # original FAIL -> mutant PASS
    #
    # original PASS -> mutant FAIL
    # --------------------------------------------------------

    f_to_p = sum(
        (
            not original
            and mutant
        )
        for original, mutant
        in zip(
            BASE_RESULTS,
            results,
        )
    )

    p_to_f = sum(
        (
            original
            and not mutant
        )
        for original, mutant
        in zip(
            BASE_RESULTS,
            results,
        )
    )


    mutant_results.append(
        {
            "id": mutant_id,
            "line": lineno,
            "kind": kind,
            "results": results,
            "f_to_p": f_to_p,
            "p_to_f": p_to_f,
        }
    )


# ============================================================
# 9. Print mutant behavior
# ============================================================

print(
    "\n================================"
)

print(
    "MUTANT RESULTS"
)

print(
    "================================"
)


for m in mutant_results:

    print(
        f"M{m['id']}: "
        f"line={m['line']} "
        f"type={m['kind']}"
    )

    print(
        "  tests:",
        [
            "P" if x else "F"
            for x in m["results"]
        ],
    )

    print(
        "  F->P:",
        m["f_to_p"],
    )

    print(
        "  P->F:",
        m["p_to_f"],
    )


# ============================================================
# 10. Compute global alpha
# ============================================================

TOTAL_F2P = sum(
    m["f_to_p"]
    for m in mutant_results
)

TOTAL_P2F = sum(
    m["p_to_f"]
    for m in mutant_results
)


alpha = (
    TOTAL_F2P * NUM_PASS
    /
    (
        TOTAL_P2F
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
    TOTAL_F2P
)

print(
    "P -> F:",
    TOTAL_P2F
)

print(
    "alpha:",
    alpha
)


# ============================================================
# 11. Group mutants by source line
# ============================================================

by_line = defaultdict(list)


for m in mutant_results:

    by_line[
        m["line"]
    ].append(m)


# ============================================================
# 12. MUSE suspiciousness
#
# mu(s)
#
# = average over mutants m of s:
#
#     F->P_m / |F|
#
#       -
#
#     alpha * P->F_m / |P|
# ============================================================

def muse_score(
    line_mutants
):

    scores = []

    for m in line_mutants:

        positive = (
            m["f_to_p"]
            / NUM_FAIL
        )

        negative = (
            alpha
            * m["p_to_f"]
            / NUM_PASS
        )

        scores.append(
            positive
            - negative
        )

    return (
        sum(scores)
        / len(scores)
    )


ranking = []


for line, mutants in by_line.items():

    score = muse_score(
        mutants
    )

    ranking.append(
        (
            score,
            line,
        )
    )


ranking.sort(
    reverse=True
)


# ============================================================
# 13. Ranking
# ============================================================

print(
    "\n================================"
)

print(
    "MUSE FAULT LOCALIZATION"
)

print(
    "================================"
)


for rank, (
    score,
    line,
) in enumerate(
    ranking,
    1,
):

    marker = (
        "  <-- BUG"
        if line == BUG_LINE
        else ""
    )

    print(
        f"{rank}. "
        f"line {line} "
        f"score={score:.3f}"
        f"{marker}"
    )


# ============================================================
# 14. Print source
# ============================================================

print(
    "\n================================"
)

print(
    "SOURCE"
)

print(
    "================================"
)


for lineno, line in enumerate(
    textwrap
    .dedent(SOURCE)
    .splitlines(),
    1,
):

    if not line:
        continue

    marker = (
        " <-- BUG"
        if lineno
        == BUG_LINE
        else ""
    )

    print(
        f"{lineno:2d}: "
        f"{line}"
        f"{marker}"
    )

