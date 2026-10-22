
import statistics
import time

import engine
from engine import (
    Analyzer,
    Function,
    Const,
    Call,
    Div,
    Return,
    BranchPositive,
)


def generate_program(n_calls):
    assert n_calls >= 2

    program = {
        "transform": Function("x", {
            "entry": (
                [],
                BranchPositive(
                    "x", "positive", "zero"
                ),
            ),
            "positive": (
                [],
                Return("x"),
            ),
            "zero": (
                [Const("result", 0)],
                Return("result"),
            ),
        }),

        "wrapper": Function("x", {
            "entry": (
                [
                    Call(
                        "r",
                        "transform",
                        "x",
                        "wrapper:transform",
                    )
                ],
                Return("r"),
            ),
        }),
    }

    main_instructions = []

    for i in range(n_calls):
        value = 10 if i == 0 else -10

        main_instructions.extend([
            Const(f"x{i}", value),
            Call(
                f"r{i}",
                "wrapper",
                f"x{i}",
                f"main:call_{i}",
            ),
        ])

    main_instructions.append(
        Div(
            "result",
            100,
            "r0",
            "main:divide",
        )
    )

    program["main"] = Function(None, {
        "entry": (
            main_instructions,
            Return("result"),
        ),
    })

    return program


def benchmark(n_calls, k, repeats=3):
    program = generate_program(n_calls)

    engine.PROGRAM.clear()
    engine.PROGRAM.update(program)

    timings = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()

        result = Analyzer(k).run()

        timings.append(
            (time.perf_counter() - start) * 1000
        )

    assert result is not None

    transform_contexts = {
        context: sorted(values)
        for (name, context), values
        in result.entries.items()
        if name == "transform"
    }

    return {
        "calls": n_calls,
        "k": k,
        "contexts": len(result.entries),
        "transform_contexts": len(
            transform_contexts
        ),
        "reanalyses": result.runs,
        "warnings": len(result.warnings),
        "false_positive": (
            "main:divide" in result.warnings
        ),
        "time_ms": statistics.median(timings),
        "inputs": transform_contexts,
    }


def main():
    print("=== CONTEXT SENSITIVITY BENCHMARK ===")

    print(
        f"{'Calls':>5} "
        f"{'k':>3} "
        f"{'Contexts':>9} "
        f"{'TransCtx':>9} "
        f"{'Runs':>7} "
        f"{'FP':>4} "
        f"{'Time(ms)':>10}"
    )

    for n in [2, 4, 8, 16]:
        for k in [0, 1, 2]:
            r = benchmark(n, k)

            print(
                f"{r['calls']:>5} "
                f"{r['k']:>3} "
                f"{r['contexts']:>9} "
                f"{r['transform_contexts']:>9} "
                f"{r['reanalyses']:>7} "
                f"{str(r['false_positive']):>4} "
                f"{r['time_ms']:>10.3f}"
            )

        print()


if __name__ == "__main__":
    main()
