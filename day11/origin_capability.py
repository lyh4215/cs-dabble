from collections import defaultdict


# ============================================================
# Tiny pseudo program
#
# 실제로는 모두 C의 FILE*라고 생각하면 됨.
#
#
# FILE *f = fopen(...);
# FILE *alias = f;
#
# fread(..., alias);
# fseek(f, ...);
# fwrite(..., f);
#
# FILE *input = stdin;
# fread(..., input);
#
# FILE *output = stdout;
# fwrite(..., output);
# ============================================================


PROGRAM = [

    # variable, origin
    (
        "origin",
        "f",
        "file"
    ),

    # alias = f
    (
        "alias",
        "alias",
        "f"
    ),

    # fread(alias)
    (
        "use",
        "alias",
        "read"
    ),

    # fseek(f)
    (
        "use",
        "f",
        "seek"
    ),

    # fwrite(f)
    (
        "use",
        "f",
        "write"
    ),


    # stdin
    (
        "origin",
        "input",
        "stdin"
    ),

    (
        "use",
        "input",
        "read"
    ),


    # stdout
    (
        "origin",
        "output",
        "stdout"
    ),

    (
        "use",
        "output",
        "write"
    ),
]


# ============================================================
# Union-Find
#
# alias 관계:
#
# alias = f
#
# 둘이 같은 stream object를 가리킬 수 있으므로
# 분석 결과를 합쳐야 한다.
# ============================================================

class UnionFind:

    def __init__(self):

        self.parent = {}


    def add(self, x):

        if x not in self.parent:
            self.parent[x] = x


    def find(self, x):

        self.add(x)

        if self.parent[x] != x:

            self.parent[x] = (
                self.find(
                    self.parent[x]
                )
            )

        return self.parent[x]


    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra


uf = UnionFind()


# ============================================================
# Pass 1
#
# 변수 수집 + alias constraint
# ============================================================

for op in PROGRAM:

    kind = op[0]

    if kind == "origin":

        _, var, _ = op

        uf.add(var)


    elif kind == "use":

        _, var, _ = op

        uf.add(var)


    elif kind == "alias":

        _, dst, src = op

        uf.union(
            dst,
            src
        )


# ============================================================
# Pass 2
#
# 각 alias group에
#
# origin
# capability
#
# constraint를 모은다.
# ============================================================

origins = defaultdict(set)
capabilities = defaultdict(set)


for op in PROGRAM:

    kind = op[0]


    if kind == "origin":

        _, var, origin = op

        root = uf.find(var)

        origins[root].add(
            origin
        )


    elif kind == "use":

        _, var, capability = op

        root = uf.find(var)

        capabilities[root].add(
            capability
        )


# ============================================================
# Group variables
# ============================================================

groups = defaultdict(list)


for var in uf.parent:

    root = uf.find(var)

    groups[root].append(
        var
    )


# ============================================================
# Rust mapping
# ============================================================

ORIGIN_TO_RUST = {

    "file":
        "std::fs::File",

    "stdin":
        "std::io::Stdin",

    "stdout":
        "std::io::Stdout",
}


CAP_TO_TRAIT = {

    "read":
        "std::io::Read",

    "write":
        "std::io::Write",

    "seek":
        "std::io::Seek",
}


# 실제 Rust 타입이 제공 가능한
# capability를 단순화한 모델

SUPPORTED = {

    "file": {
        "read",
        "write",
        "seek"
    },

    "stdin": {
        "read"
    },

    "stdout": {
        "write"
    }
}


# ============================================================
# Print
# ============================================================

print(
    "=== ORIGIN / CAPABILITY ANALYSIS ==="
)

print()


for root, variables in groups.items():

    origin = origins[root]
    caps = capabilities[root]


    print(
        f"variables     : "
        f"{', '.join(variables)}"
    )

    print(
        f"origin        : "
        f"{', '.join(sorted(origin))}"
    )

    print(
        f"capabilities  : "
        f"{', '.join(sorted(caps))}"
    )


    # --------------------------------------------------------
    # Origin이 정확히 하나면
    # Rust concrete type 추론 가능
    # --------------------------------------------------------

    if len(origin) == 1:

        o = next(
            iter(origin)
        )

        rust_type = (
            ORIGIN_TO_RUST[o]
        )

        print(
            f"Rust type     : "
            f"{rust_type}"
        )


        traits = [
            CAP_TO_TRAIT[c]
            for c in sorted(caps)
        ]

        print(
            f"required trait: "
            f"{', '.join(traits)}"
        )


        unsupported = (
            caps
            - SUPPORTED[o]
        )


        if unsupported:

            print(
                "STATUS        : INVALID"
            )

            print(
                "unsupported   : "
                + ", ".join(
                    unsupported
                )
            )

        else:

            print(
                "STATUS        : OK"
            )


    else:

        print(
            "Rust type     : "
            "AMBIGUOUS"
        )

        print(
            "STATUS        : "
            "needs more analysis"
        )


    print()