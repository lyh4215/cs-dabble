import random
from collections import defaultdict


# ============================================================
# 1. TOY REPOSITORY
#
# 실제 root cause는 cache.cpp에 있지만
# crash stack에는 cache.cpp가 나타나지 않는다.
# ============================================================

REPO = {

    "decoder.cpp": {
        91: """
Buffer *buf = parser.get_buffer(key);

// crash happens here
return buf->data[0];
"""
    },

    "parser.cpp": {
        42: """
Buffer *Parser::get_buffer(Key key) {
    return lookup_buffer(key);
}
"""
    },

    "cache.cpp": {
        73: """
Buffer *lookup_buffer(Key key) {

    if (!cache.contains(key)) {

        // BUG:
        // fallback buffer를 반환해야 하는데
        // nullptr를 반환함.
        return nullptr;
    }

    return cache[key];
}
"""
    },
}


ROOT_CAUSE = "cache.cpp"


# ============================================================
# 2. CRASH DUMP
# ============================================================

CRASH_INFO = {
    "signal": "SIGSEGV",
    "fault_address": "0x0",
}


CRASH_STACK = [

    {
        "file": "decoder.cpp",
        "line": 91,
        "function": "Decoder::decode",
    },

    {
        "file": "parser.cpp",
        "line": 42,
        "function": "Parser::get_buffer",
    },
]


# ============================================================
# 3. DEFINITION DATABASE
#
# 실제 대규모 repository라면 clangd / LSP 등을 통해
# identifier definition을 찾을 수 있다.
#
# 여기서는 mapping으로 압축.
# ============================================================

DEFINITIONS = {

    (
        "parser.cpp",
        "lookup_buffer",
    ): (
        "cache.cpp",
        73,
    ),
}


# ============================================================
# 4. TOOLS
#
# agent가 직접 repository 전체를 context에 넣는 게 아니라
# 필요한 정보만 tool을 통해 요청한다고 생각.
# ============================================================

def get_crash_info():

    return CRASH_INFO


def get_crash_stack():

    return CRASH_STACK


def get_nearby_code(
    file,
    line,
):

    return REPO[
        file
    ].get(
        line,
        "<code unavailable>",
    )


def get_term_definition(
    file,
    identifier,
):

    return DEFINITIONS.get(
        (
            file,
            identifier,
        )
    )


# ============================================================
# 5. ONE AGENT RUN
#
# 실제 LLM은 아니고,
# AutoCrashFL의 탐색 구조만 재현하는 heuristic agent.
#
#
# crash
#   ↓
# stack
#   ↓
# nearby code
#   ↓
# identifier 발견
#   ↓
# definition navigation
#   ↓
# stack 밖 file 탐색
# ============================================================

def run_agent(
    run_id,
    deep_search=True,
):

    rng = random.Random(
        run_id
    )


    trace = []

    suspicious = set()


    # --------------------------------------------------------
    # Step 1: crash info
    # --------------------------------------------------------

    info = get_crash_info()

    trace.append(
        (
            "get_crash_info",
            info,
        )
    )


    # --------------------------------------------------------
    # Step 2: stack
    # --------------------------------------------------------

    stack = get_crash_stack()

    trace.append(
        (
            "get_crash_stack",
            stack,
        )
    )


    # --------------------------------------------------------
    # Step 3: crash site
    # --------------------------------------------------------

    top = stack[0]

    top_code = get_nearby_code(
        top["file"],
        top["line"],
    )

    trace.append(
        (
            "get_nearby_code",
            top["file"],
        )
    )


    # SIGSEGV가 실제로 발생한 위치
    suspicious.add(
        "decoder.cpp"
    )


    # --------------------------------------------------------
    # Step 4: caller
    # --------------------------------------------------------

    caller = stack[1]

    caller_code = get_nearby_code(
        caller["file"],
        caller["line"],
    )

    trace.append(
        (
            "get_nearby_code",
            caller["file"],
        )
    )


    # LLM이 caller를 의심하는 상황을 흉내
    if rng.random() < 0.85:

        suspicious.add(
            "parser.cpp"
        )


    # ========================================================
    # Step 5: deep repository navigation
    #
    # parser.cpp에 있는:
    #
    #     lookup_buffer(key)
    #
    # definition을 찾아간다.
    # ========================================================

    if deep_search:

        follow_definition = (
            rng.random()
            < 0.75
        )


        if follow_definition:

            location = (
                get_term_definition(
                    "parser.cpp",
                    "lookup_buffer",
                )
            )


            trace.append(
                (
                    "get_term_definition",
                    location,
                )
            )


            if location:

                file, line = (
                    location
                )


                root_code = (
                    get_nearby_code(
                        file,
                        line,
                    )
                )


                trace.append(
                    (
                        "get_nearby_code",
                        file,
                    )
                )


                # ------------------------------------------------
                # 실제 reasoning은 LLM이 하겠지만,
                # toy에서는 nullptr return을 강한 signal로 사용.
                #
                # decoder.cpp:
                #
                #     buf->data
                #
                # cache.cpp:
                #
                #     return nullptr
                #
                # 두 조각을 연결.
                # ------------------------------------------------

                if (
                    "return nullptr"
                    in root_code
                ):

                    suspicious.add(
                        "cache.cpp"
                    )


                    # 어떤 run은 root cause만 강하게 선택
                    if rng.random() < 0.55:

                        suspicious = {
                            "cache.cpp"
                        }

                    else:

                        # crash manifestation보다
                        # upstream 후보를 더 중요하게 본다.
                        suspicious.discard(
                            "decoder.cpp"
                        )


    # --------------------------------------------------------
    # 일부 run은 reasoning이 실패해서
    # crash site에 고착된다고 가정.
    # --------------------------------------------------------

    if rng.random() < 0.10:

        suspicious = {
            "decoder.cpp"
        }


    return {
        "run": run_id,
        "suspicious": suspicious,
        "trace": trace,
    }


# ============================================================
# 6. MULTI-RUN AGGREGATION
#
# 한 run이 suspicious set S를 반환하면:
#
# 각 file에 1 / |S|
#
# 만큼 score를 준다.
#
# 예:
#
# {cache.cpp}
#
# cache += 1
#
#
# {cache.cpp, parser.cpp}
#
# cache  += 0.5
# parser += 0.5
# ============================================================

def aggregate(
    runs,
):

    scores = defaultdict(
        float
    )


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


    ranking = sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True,
    )


    return ranking


# ============================================================
# 7. Confidence
#
# top file이 전체 R번 run 중
# 얼마나 강하게 반복 지지받았는가?
# ============================================================

def confidence(
    ranking,
    R,
):

    if not ranking:

        return 0.0


    top_score = (
        ranking[0][1]
    )


    return (
        top_score / R
    )


# ============================================================
# 8. STACK-ONLY BASELINE
#
# stack trace에 나온 파일만 후보로 삼는다.
# ============================================================

def stack_baseline():

    seen = set()

    result = []


    for frame in CRASH_STACK:

        file = frame[
            "file"
        ]


        if file not in seen:

            seen.add(
                file
            )

            result.append(
                file
            )


    return result


# ============================================================
# 9. EXPERIMENT
# ============================================================

def experiment(
    R=10,
    deep_search=True,
):

    runs = [

        run_agent(
            run_id=i,
            deep_search=deep_search,
        )

        for i in range(R)
    ]


    # --------------------------------------------------------
    # stack baseline
    # --------------------------------------------------------

    print(
        "================================"
    )

    print(
        "STACK BASELINE"
    )

    print(
        "================================"
    )


    baseline = (
        stack_baseline()
    )


    for rank, file in enumerate(
        baseline,
        1,
    ):

        marker = (

            " <-- ROOT CAUSE"

            if file
            == ROOT_CAUSE

            else ""
        )


        print(
            f"{rank}. "
            f"{file}"
            f"{marker}"
        )


    print(
        "\nroot cause in stack:",
        ROOT_CAUSE
        in baseline,
    )


    # --------------------------------------------------------
    # individual runs
    # --------------------------------------------------------

    print(
        "\n================================"
    )

    print(
        "INDEPENDENT RUNS"
    )

    print(
        "================================"
    )


    for r in runs:

        print(
            f"run {r['run']:2d}: "
            f"{sorted(r['suspicious'])}"
        )


    # --------------------------------------------------------
    # aggregation
    # --------------------------------------------------------

    ranking = aggregate(
        runs
    )


    conf = confidence(
        ranking,
        R,
    )


    print(
        "\n================================"
    )

    print(
        "AGGREGATED RANKING"
    )

    print(
        "================================"
    )


    for rank, (
        file,
        score,
    ) in enumerate(
        ranking,
        1,
    ):

        marker = (

            " <-- ROOT CAUSE"

            if file
            == ROOT_CAUSE

            else ""
        )


        print(
            f"{rank}. "
            f"{file:12s} "
            f"score={score:.3f}"
            f"{marker}"
        )


    print(
        "\nconfidence:",
        round(
            conf,
            3,
        )
    )


    if ranking:

        print(
            "top-1 correct:",
            ranking[0][0]
            == ROOT_CAUSE,
        )


    # --------------------------------------------------------
    # successful deep-search trace
    # --------------------------------------------------------

    if deep_search:

        print(
            "\n================================"
        )

        print(
            "EXAMPLE SEARCH TRACE"
        )

        print(
            "================================"
        )


        for r in runs:

            if (
                ROOT_CAUSE
                in r["suspicious"]
            ):

                for (
                    step,
                    result,
                ) in r["trace"]:

                    print(
                        f"{step:22s}"
                        f" -> {result}"
                    )

                break


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "################################"
    )

    print(
        "WITH DEEP SEARCH"
    )

    print(
        "################################"
    )

    experiment(
        R=10,
        deep_search=True,
    )


    print(
        "\n\n"
        "################################"
    )

    print(
        "WITHOUT DEEP SEARCH"
    )

    print(
        "################################"
    )

    experiment(
        R=10,
        deep_search=False,
    )