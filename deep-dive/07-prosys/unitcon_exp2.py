
from collections import Counter, deque
from dataclasses import dataclass, replace
import time

from unitcon_exp1 import verify_java


ACTIONS = (
    "CREATE", "RESET", "REVOKE", "RESTORE",
    "DISABLE", "PING", "AUTHORIZE",
    "ENABLE", "NULLIFY",
)


@dataclass(frozen=True)
class State:
    exists: bool = False
    authorized: bool = False
    enabled: bool = False
    payload_null: bool = False
    noise: int = 0


def step(s: State, action: str) -> State:
    if action == "CREATE":
        return State(exists=True)

    if not s.exists:
        return s

    if action == "RESET":
        return State(exists=True)

    if action == "REVOKE":
        return replace(
            s, authorized=False, enabled=False
        )

    if action == "RESTORE":
        return replace(s, payload_null=False)

    if action == "DISABLE":
        return replace(s, enabled=False)

    if action == "PING":
        return replace(
            s, noise=(s.noise + 1) % 16
        )

    if action == "AUTHORIZE":
        return replace(s, authorized=True)

    if action == "ENABLE" and s.authorized:
        return replace(s, enabled=True)

    if action == "NULLIFY" and s.enabled:
        return replace(s, payload_null=True)

    return s


def target(s: State) -> bool:
    return (
        s.exists
        and s.enabled
        and s.payload_null
    )


def abstract_key(s: State, mode: str):
    if mode == "full":
        return s

    if mode == "relevant":
        return (
            s.exists,
            s.authorized,
            s.enabled,
            s.payload_null,
        )

    if mode == "unsafe":
        # Deliberately forgets 'enabled'
        return (
            s.exists,
            s.authorized,
            s.payload_null,
        )

    raise ValueError(mode)


def synthesize(mode: str, max_depth=7):
    initial = State()
    queue = deque([((), initial)])

    visited = set()

    if mode != "none":
        visited.add(abstract_key(initial, mode))

    explored = 0
    generated = 0
    pruned = 0
    by_depth = Counter()

    start = time.perf_counter()

    while queue:
        sequence, state = queue.popleft()

        explored += 1
        by_depth[len(sequence)] += 1

        if target(state):
            elapsed = (
                time.perf_counter() - start
            ) * 1000

            return {
                "witness": sequence,
                "explored": explored,
                "generated": generated,
                "pruned": pruned,
                "elapsed_ms": elapsed,
                "by_depth": dict(by_depth),
            }

        if len(sequence) >= max_depth:
            continue

        for action in ACTIONS:
            generated += 1

            new_state = step(state, action)
            new_sequence = sequence + (action,)

            if mode != "none":
                k = abstract_key(
                    new_state, mode
                )

                if k in visited:
                    pruned += 1
                    continue

                visited.add(k)

            queue.append((
                new_sequence,
                new_state,
            ))

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return {
        "witness": None,
        "explored": explored,
        "generated": generated,
        "pruned": pruned,
        "elapsed_ms": elapsed,
        "by_depth": dict(by_depth),
    }


def main():
    results = {}

    for mode in (
        "none",
        "full",
        "relevant",
        "unsafe",
    ):
        r = synthesize(mode)
        results[mode] = r

        print(f"\n=== {mode.upper()} ===")
        print("Explored:", r["explored"])
        print("Generated:", r["generated"])
        print("Pruned:", r["pruned"])
        print("Elapsed ms:", round(r["elapsed_ms"], 3))
        print("Witness:", r["witness"])

        if r["witness"] is not None:
            actual = verify_java(r["witness"])
            print("JVM:", actual)
            assert actual == "TARGET_NPE"

    assert (
        results["none"]["witness"]
        == results["full"]["witness"]
        == results["relevant"]["witness"]
    )

    assert results["unsafe"]["witness"] is None

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
