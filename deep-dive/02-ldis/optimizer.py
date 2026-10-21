
from collections import Counter


def make_tables():
    return {
        "A": Counter({0: 900, 1: 100}),
        "B": Counter({0: 900, 1: 100}),
        "C": Counter({0: 100, 1: 900}),
    }


def actual_join(r, s):
    return Counter({
        k: r[k] * s[k]
        for k in r.keys() | s.keys()
        if r[k] * s[k] > 0
    })


def actual_cardinality(r, s):
    return sum(actual_join(r, s).values())


def estimate_join(r, s):
    nr = sum(r.values())
    ns = sum(s.values())

    vr = len(r)
    vs = len(s)

    return nr * ns / max(vr, vs)


def evaluate_plans(tables):
    a, b, c = (tables[x] for x in "ABC")

    ab = actual_join(a, b)
    bc = actual_join(b, c)

    # Both plans produce the same final result.
    final_ab = actual_cardinality(ab, c)
    final_bc = actual_cardinality(a, bc)
    assert final_ab == final_bc

    plans = {
        "AB_FIRST": {
            "estimated": estimate_join(a, b),
            "actual": sum(ab.values()),
        },
        "BC_FIRST": {
            "estimated": estimate_join(b, c),
            "actual": sum(bc.values()),
        },
    }

    # Simplified cost model:
    # both plans have identical scans and final output.
    # Compare intermediate cardinality only.
    chosen = min(
        plans,
        key=lambda p: plans[p]["estimated"]
    )

    optimal = min(
        plans,
        key=lambda p: plans[p]["actual"]
    )

    return plans, chosen, optimal


if __name__ == "__main__":
    tables = make_tables()

    plans, chosen, optimal = evaluate_plans(tables)

    for name, info in plans.items():
        print(f"\n{name}")
        print(f"Estimated: {info['estimated']:,.0f}")
        print(f"Actual:    {info['actual']:,.0f}")

    print(f"\nOptimizer selected: {chosen}")
    print(f"Actual best plan:   {optimal}")

    regret = (
        plans[chosen]["actual"]
        / plans[optimal]["actual"]
    )
    print(f"Intermediate cost ratio: {regret:.2f}x")
