
"""ProSys UnitCon Exp 5: flow-sensitive MAY points-to + nullness analysis.

Small acyclic CFG DSL. Allocation-site abstraction; no loops, fields of Box only.
Uses join at control-flow merges, and strong/weak heap updates.
Python 3.11+, no dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque


Obj = str
Ref = str


@dataclass(frozen=True)
class Node:
    op: str
    x: str = ""
    y: str = ""
    succ: tuple[int, ...] = ()


@dataclass(frozen=True)
class Abs:
    refs: tuple[tuple[Ref, frozenset[Obj]], ...] = ()
    heap: tuple[tuple[Obj, frozenset[bool]], ...] = ()

    def refmap(self):
        return dict(self.refs)

    def heapmap(self):
        return dict(self.heap)


def abstract(
    refs: dict[str, set[str]],
    heap: dict[str, set[bool]],
) -> Abs:
    return Abs(
        tuple(
            sorted(
                (k, frozenset(v))
                for k, v in refs.items()
            )
        ),
        tuple(
            sorted(
                (k, frozenset(v))
                for k, v in heap.items()
            )
        ),
    )


def join(a: Abs, b: Abs) -> Abs:
    ar, br = a.refmap(), b.refmap()
    ah, bh = a.heapmap(), b.heapmap()

    return abstract(
        {
            k: (
                set(ar.get(k, frozenset()))
                | set(br.get(k, frozenset()))
            )
            for k in ar.keys() | br.keys()
        },
        {
            k: (
                set(ah.get(k, frozenset()))
                | set(bh.get(k, frozenset()))
            )
            for k in ah.keys() | bh.keys()
        },
    )


def transfer(node: Node, state: Abs) -> Abs:
    refs = {
        k: set(v)
        for k, v in state.refmap().items()
    }

    heap = {
        k: set(v)
        for k, v in state.heapmap().items()
    }

    if node.op == "NEW":
        refs[node.x] = {node.y}
        heap[node.y] = {False}  # Box.value = "ready"

    elif node.op == "COPY":
        refs[node.x] = set(
            refs.get(node.y, set())
        )

    elif node.op == "STORE_NULL":
        targets = refs.get(node.x, set())

        for obj in targets:
            if len(targets) == 1:
                # Strong update
                heap[obj] = {True}
            else:
                # Weak update
                heap.setdefault(obj, set()).add(True)

    elif node.op not in ("BRANCH", "TARGET"):
        raise ValueError(node.op)

    return abstract(refs, heap)


def analyze(cfg: dict[int, Node]) -> dict[int, Abs]:
    # IN states of each node.
    # Propagate states until a fixed point.

    incoming = {0: Abs()}
    queue = deque([0])

    while queue:
        pc = queue.popleft()

        after = transfer(
            cfg[pc],
            incoming[pc],
        )

        for nxt in cfg[pc].succ:
            if nxt in incoming:
                updated = join(
                    incoming[nxt],
                    after,
                )
            else:
                updated = after

            if (
                nxt not in incoming
                or updated != incoming[nxt]
            ):
                incoming[nxt] = updated
                queue.append(nxt)

    return incoming


def concrete_paths(cfg: dict[int, Node]):
    # Execute every branch of the acyclic CFG.

    work = [(0, {}, {}, ())]
    results = []

    while work:
        pc, refs, heap, path = work.pop()

        node = cfg[pc]

        refs = dict(refs)
        heap = dict(heap)

        if node.op == "NEW":
            refs[node.x] = node.y
            heap[node.y] = False

        elif node.op == "COPY":
            refs[node.x] = refs[node.y]

        elif node.op == "STORE_NULL":
            heap[refs[node.x]] = True

        path = path + (pc,)

        if node.op == "TARGET":
            results.append(
                (
                    path,
                    heap[refs[node.x]],
                    dict(refs),
                    dict(heap),
                )
            )
        else:
            for nxt in reversed(node.succ):
                work.append(
                    (nxt, refs, heap, path)
                )

    return results


def cases():
    return {
        "must_alias": {
            0: Node("NEW", "a", "O1", (1,)),
            1: Node("COPY", "b", "a", (2,)),
            2: Node("STORE_NULL", "b", succ=(3,)),
            3: Node("TARGET", "a"),
        },

        "no_alias": {
            0: Node("NEW", "a", "O1", (1,)),
            1: Node("NEW", "b", "O2", (2,)),
            2: Node("STORE_NULL", "b", succ=(3,)),
            3: Node("TARGET", "a"),
        },

        "may_alias": {
            0: Node("NEW", "a", "O1", (1,)),
            1: Node("BRANCH", succ=(2, 3)),
            2: Node("COPY", "b", "a", (4,)),
            3: Node("NEW", "b", "O2", (4,)),
            4: Node("STORE_NULL", "b", succ=(5,)),
            5: Node("TARGET", "a"),
        },
    }


def check(cfg, incoming):
    target_pc = next(
        i
        for i, node in cfg.items()
        if node.op == "TARGET"
    )

    state = incoming[target_pc]

    a = state.refmap().get(
        "a", frozenset()
    )

    b = state.refmap().get(
        "b", frozenset()
    )

    may_alias = bool(a & b)

    must_alias = (
        len(a) == len(b) == 1
        and a == b
    )

    nullness = set().union(
        *(
            state.heapmap().get(
                obj, frozenset()
            )
            for obj in a
        )
    )

    paths = concrete_paths(cfg)

    actual = {
        npe
        for _, npe, _, _ in paths
    }

    assert actual <= nullness, (
        actual,
        nullness,
    )

    return (
        a,
        b,
        may_alias,
        must_alias,
        nullness,
        paths,
    )


def main():
    for name, cfg in cases().items():
        incoming = analyze(cfg)

        (
            a,
            b,
            may,
            must,
            nullness,
            paths,
        ) = check(cfg, incoming)

        print(f"\n[{name}]")

        print(
            "  points-to(a):", sorted(a),
            "points-to(b):", sorted(b),
        )

        print(
            "  may-alias:", may,
            "must-alias:", must,
        )

        print(
            "  possible a.value nullness:",
            sorted(nullness),
        )

        print(
            "  concrete paths:",
            [
                (
                    path,
                    "NPE" if npe else "SAFE",
                )
                for path, npe, _, _ in paths
            ],
        )

        if name == "may_alias":
            assert may and not must
            assert nullness == {False, True}

            assert {
                n for _, n, _, _ in paths
            } == {False, True}

            print(
                "  WARNING: Assuming b does not "
                "alias a would miss the NPE branch!"
            )

        elif name == "must_alias":
            assert may and must
            assert nullness == {True}

        else:
            assert not may and not must
            assert nullness == {False}

    print(
        "\nAll analysis/concrete cross-checks passed."
    )


if __name__ == "__main__":
    main()
