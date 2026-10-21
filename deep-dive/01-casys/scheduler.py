
from dataclasses import dataclass
import random
import statistics

@dataclass
class Request:
    id: int
    arrival: float
    prompt: int
    output: int
    remaining: int = 0
    start: float | None = None
    finish: float | None = None

    def __post_init__(self):
        self.remaining = self.output



def workload(n=100, seed=42, mean_interarrival=5.0):
    rng = random.Random(seed)
    t = 0.0
    requests = []

    for i in range(n):
        t += rng.expovariate(1 / mean_interarrival)
        requests.append(Request(
            id=i,
            arrival=t,
            prompt=rng.choice([32, 64, 128, 256]),
            output=rng.choice([16, 32, 64, 128]),
        ))

    return requests


def simulate(requests, policy, max_batch=8):
    assert policy in ("static", "continuous")

    pending = list(requests)
    active = []
    completed = []
    t = 0.0

    while pending or active:
        if not active and pending and pending[0].arrival > t:
            t = pending[0].arrival

        # Static: admit only when the previous batch completes.
        # Continuous: refill available slots at each step.
        can_admit = policy == "continuous" or not active

        if can_admit:
            while pending and len(active) < max_batch:
                if pending[0].arrival > t:
                    break

                r = pending.pop(0)
                r.start = t

                # Simplified sequential prefill cost (ms).
                t += 0.02 * r.prompt
                active.append(r)

        if not active:
            if pending:
                t = max(t, pending[0].arrival)
            continue

        # All active requests generate one token per step.
        # Simplified GPU decode cost (ms).
        step_time = 0.6 + 0.12 * len(active)
        t += step_time

        survivors = []
        for r in active:
            r.remaining -= 1
            if r.remaining == 0:
                r.finish = t
                completed.append(r)
            else:
                survivors.append(r)

        active = survivors

    latency = [r.finish - r.arrival for r in completed]
    duration = max(r.finish for r in completed) - min(
        r.arrival for r in requests
    )

    return {
        "throughput": len(completed) / (duration / 1000),
        "mean_latency": statistics.mean(latency),
        "p99_latency": sorted(latency)[
            max(0, int(0.99 * len(latency)) - 1)
        ],
    }


if __name__ == "__main__":
    for policy in ("static", "continuous"):
        result = simulate(workload(), policy)
        print(f"\n{policy.upper()}")
        for k, v in result.items():
            print(f"  {k:16s}: {v:.2f}")
