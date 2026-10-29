
import argparse
import random
from collections import deque
from dataclasses import dataclass

DT = 0.02
PERIOD = 0.20
DEADLINE = 0.15
LIMIT = 55.0
RECOVER = 51.0


@dataclass
class Job:
    release: float
    remaining: float
    deadline: float
    missed: bool = False


def simulate(
    policy: str,
    ambient: float,
    seed: int,
    seconds: int = 90,
):
    rng = random.Random(seed)

    queue = deque()
    temp = ambient
    throttled = False

    releases = 0
    missed = 0
    completed = 0
    energy = 0.0
    freq_time = 0.0
    max_temp = temp

    traces = []

    steps = int(seconds / DT)

    for k in range(steps):
        now = k * DT

        # Periodic job release
        if k % round(PERIOD / DT) == 0:
            demand = (
                0.22
                if (k // 10) % 30 < 18
                else 0.29
            )

            demand *= rng.uniform(0.93, 1.07)

            queue.append(
                Job(
                    release=now,
                    remaining=demand,
                    deadline=now + DEADLINE,
                )
            )

            releases += 1

        # DVFS governor
        if policy == "performance":
            requested = 2.0

        elif policy == "reactive":
            requested = (
                1.0 if temp >= 51 else 2.0
            )

        elif policy == "proactive":
            requested = (
                1.5 if temp >= 46 else 2.0
            )

        elif policy == "fixed_mid":
            requested = 1.5

        else:
            raise ValueError(policy)

        # Thermal emergency throttle
        if temp >= LIMIT:
            throttled = True

        elif temp <= RECOVER:
            throttled = False

        freq = 1.0 if throttled else requested

        # Check deadline misses
        for job in queue:
            if (
                not job.missed
                and now + 1e-9 >= job.deadline
                and job.remaining > 1e-9
            ):
                job.missed = True
                missed += 1

        # Execute jobs in FIFO order
        capacity = freq * DT

        while queue and capacity > 1e-10:
            job = queue[0]

            executed = min(
                job.remaining,
                capacity,
            )

            job.remaining -= executed
            capacity -= executed

            if job.remaining < 1e-9:
                finish = (
                    now + DT - capacity / freq
                )

                if (
                    finish > job.deadline + 1e-9
                    and not job.missed
                ):
                    job.missed = True
                    missed += 1

                queue.popleft()
                completed += 1
            else:
                break

        # Simplified power model
        dynamic = 2.5 * (freq / 2) ** 3
        power = 0.8 + dynamic

        energy += power * DT
        freq_time += freq * DT

        # Simplified RC thermal model
        temp += DT * (
            0.85 * power
            - 0.075 * (temp - ambient)
        )

        max_temp = max(max_temp, temp)

        if k % 50 == 0:
            traces.append((
                round(now, 2),
                round(temp, 2),
                freq,
                len(queue),
            ))

    return {
        "policy": policy,
        "ambient": ambient,
        "seed": seed,
        "jobs": releases,
        "completed": completed,
        "miss_pct": round(
            100 * missed / releases, 2
        ),
        "energy_j": round(energy, 2),
        "max_temp_c": round(max_temp, 2),
        "avg_freq": round(
            freq_time / seconds, 3
        ),
        "trace": traces,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seconds",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    print(
        f"{'ambient':>7} "
        f"{'policy':>12} "
        f"{'miss%':>9} "
        f"{'energy J':>11} "
        f"{'max temp':>10}"
    )

    policies = (
        "performance",
        "reactive",
        "proactive",
        "fixed_mid",
    )

    for ambient in (20.0, 30.0, 40.0):
        for policy in policies:
            results = [
                simulate(
                    policy,
                    ambient,
                    seed,
                    args.seconds,
                )
                for seed in range(args.seeds)
            ]

            def avg(key):
                return sum(
                    result[key]
                    for result in results
                ) / len(results)

            print(
                f"{ambient:7.0f} "
                f"{policy:>12} "
                f"{avg('miss_pct'):9.2f} "
                f"{avg('energy_j'):11.1f} "
                f"{avg('max_temp_c'):10.2f}"
            )


if __name__ == "__main__":
    main()
