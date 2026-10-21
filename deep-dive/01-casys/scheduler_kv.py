
from dataclasses import dataclass, replace
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

    @property
    def kv_size(self):
        # Conservative reservation: entire future KV cache.
        return self.prompt + self.output


def workload(n=500, seed=42, mean_interarrival=10):
    rng = random.Random(seed)
    t = 0.0
    requests = []

    for i in range(n):
        t += rng.expovariate(1 / mean_interarrival)

        requests.append(Request(
            id=i,
            arrival=t,
            prompt=rng.choice([32, 64, 128, 256]),
            output=rng.choice([16, 32, 64, 128, 256]),
        ))

    return requests


def percentile(values, pct):
    values = sorted(values)
    index = max(0, int((len(values) - 1) * pct))
    return values[index]


def simulate(requests, policy, kv_capacity=1024,
             max_batch=16, alpha=0.0):

    assert policy in ("fcfs", "backfill", "shortest", "aging")

    # Clone all requests: policies receive identical workloads.
    pending = sorted(
        [replace(r, remaining=r.output,
                 start=None, finish=None) for r in requests],
        key=lambda r: (r.arrival, r.id)
    )

    if any(r.kv_size > kv_capacity for r in pending):
        raise ValueError("Request exceeds KV capacity")

    waiting = []
    active = []
    completed = []

    t = 0.0
    used_kv = 0

    max_used_kv = 0
    utilization_sum = 0.0
    elapsed_sum = 0.0

    def advance(dt):
        nonlocal t, utilization_sum, elapsed_sum
        utilization_sum += used_kv / kv_capacity * dt
        elapsed_sum += dt
        t += dt

    while pending or waiting or active:

        # When GPU is idle, advance to the next arrival.
        if not waiting and not active and pending:
            if pending[0].arrival > t:
                advance(pending[0].arrival - t)

        # Collect all requests that have arrived.
        while pending and pending[0].arrival <= t:
            waiting.append(pending.pop(0))

        # Admission control.
        while waiting and len(active) < max_batch:

            available = kv_capacity - used_kv

            
            if policy == "fcfs":
                candidate = waiting[0]
                if candidate.kv_size > available:
                    break

            else:
                fitting = [
                    r for r in waiting
                    if r.kv_size <= available
                ]

                if not fitting:
                    break

                if policy == "backfill":
                    candidate = min(
                        fitting,
                        key=lambda r: (r.arrival, r.id)
                    )

                elif policy == "aging":
                    candidate = min(
                        fitting,
                        key=lambda r: (
                            r.output - alpha * (t - r.arrival),
                            r.arrival,
                            r.id,
                        )
                    )

                elif policy == "shortest":
                    candidate = min(
                        fitting,
                        key=lambda r: (
                            r.output,
                            r.kv_size,
                            r.arrival,
                            r.id,
                        )
                    )


            waiting.remove(candidate)
            candidate.start = t

            used_kv += candidate.kv_size
            max_used_kv = max(max_used_kv, used_kv)
            active.append(candidate)

            # Simplified serialized prefill cost.
            advance(0.02 * candidate.prompt)

        if not active:
            if pending:
                advance(max(0, pending[0].arrival - t))
                continue

            if waiting:
                raise RuntimeError("No admissible request")
            break

        # One decode token per active request.
        step_time = 0.6 + 0.12 * len(active)
        advance(step_time)

        survivors = []

        for r in active:
            r.remaining -= 1

            if r.remaining == 0:
                r.finish = t
                used_kv -= r.kv_size
                completed.append(r)
            else:
                survivors.append(r)

        active = survivors

    latencies = [
        r.finish - r.arrival for r in completed
    ]

    long_requests = [
        r for r in completed if r.output >= 128
    ]

    long_latencies = [
        r.finish - r.arrival for r in long_requests
    ]

    duration = (
        max(r.finish for r in completed)
        - min(r.arrival for r in requests)
    )

    waiting_times = [
        r.start - r.arrival
        for r in completed
    ]

    long_waiting_times = [
        r.start - r.arrival
        for r in completed
        if r.output >= 128
    ]

    return {
        "throughput": len(completed) / (duration / 1000),
        "mean_latency": statistics.mean(latencies),
        "p99_latency": percentile(latencies, 0.99),
        "long_mean_latency": statistics.mean(long_latencies),
        "max_latency": max(latencies),
        "avg_kv_usage": utilization_sum / elapsed_sum,
        "peak_kv_usage": max_used_kv / kv_capacity,
        "mean_waiting": statistics.mean(waiting_times),
        "max_waiting": max(waiting_times),
        "long_p99_waiting": percentile(long_waiting_times, 0.99),
    }


if __name__ == "__main__":
    requests = workload()

    for policy in ("fcfs", "backfill", "shortest"):
        result = simulate(requests, policy)

        print(f"\n{policy.upper()}")
        for k, v in result.items():
            print(f"{k:20s}: {v:.3f}")

    for aging in [0.00, 0.01, 0.05, 0.10, 0.20]:
        result = simulate(requests, "aging", alpha=aging)

        print(f"\nAGING, {aging=}")
        for k, v in result.items():
            print(f"{k:20s}: {v:.3f}")
