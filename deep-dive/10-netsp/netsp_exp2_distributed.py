
from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    prefix: str
    origin: int


@dataclass(frozen=True)
class Update:
    at_ms: int
    router: int
    route: Route
    authorized: bool


class Owner:
    def __init__(self, authorized: set[Route]):
        self.authorized = authorized
        self.queries = 0

    def verify(self, route: Route) -> bool:
        self.queries += 1
        return route in self.authorized


def workload(
    routers: int,
    attacks_per_router: int,
    legit_per_router: int,
    seed: int,
) -> list[Update]:
    rng = random.Random(seed)
    valid = Route("203.0.113.0/24", 64501)
    result = []

    for r in range(routers):
        for j in range(attacks_per_router):
            attacker = Route(
                "203.0.113.0/24",
                65000 + r * attacks_per_router + j,
            )
            result.append(
                Update(
                    rng.randrange(1000),
                    r,
                    attacker,
                    False,
                )
            )

        for _ in range(legit_per_router):
            result.append(
                Update(
                    rng.randrange(1000),
                    r,
                    valid,
                    True,
                )
            )

    return sorted(
        result,
        key=lambda x: (
            x.at_ms,
            x.router,
            x.route.origin,
        ),
    )


def run(
    updates: list[Update],
    routers: int,
    local_budget: int,
    global_budget: int,
    strategy: str,
) -> dict:
    owner = Owner({
        Route("203.0.113.0/24", 64501)
    })

    local = [0] * routers
    central_count = 0

    proof_cache: set[Route] = set()
    decision = Counter()
    requests_by_router = Counter()

    for update in updates:
        # Cached owner approval can bypass new queries.
        if (
            strategy == "global-cache"
            and update.route in proof_cache
        ):
            decision[
                (update.authorized, True)
            ] += 1
            continue

        # Per-router query budget
        if local[update.router] >= local_budget:
            decision[
                (update.authorized, False)
            ] += 1
            continue

        # Shared global query budget
        if (
            strategy != "local"
            and central_count >= global_budget
        ):
            decision[
                (update.authorized, False)
            ] += 1
            continue

        local[update.router] += 1

        if strategy != "local":
            central_count += 1

        requests_by_router[update.router] += 1

        accepted = owner.verify(update.route)

        if strategy == "global-cache" and accepted:
            proof_cache.add(update.route)

        decision[
            (update.authorized, accepted)
        ] += 1

    legitimate = sum(
        update.authorized for update in updates
    )

    return {
        "strategy": strategy,
        "owner_queries": owner.queries,
        "legit_accepted": decision[(True, True)],
        "legit_total": legitimate,
        "attacker_accepted": decision[(False, True)],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--routers", type=int, default=20
    )
    parser.add_argument(
        "--attacks-per-router",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--legit-per-router",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--local-budget", type=int, default=10
    )
    parser.add_argument(
        "--global-budget", type=int, default=50
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    updates = workload(
        args.routers,
        args.attacks_per_router,
        args.legit_per_router,
        args.seed,
    )

    print("Total updates:", len(updates))

    for strategy in (
        "local",
        "global",
        "global-cache",
    ):
        result = run(
            updates,
            args.routers,
            args.local_budget,
            args.global_budget,
            strategy,
        )

        print(strategy, result)


if __name__ == "__main__":
    main()
