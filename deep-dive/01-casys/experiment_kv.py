
from scheduler_kv import workload, simulate
import statistics

CAPACITIES = [512, 768, 1024, 2048, 4096]
ARRIVALS = [5, 10, 20]
SEEDS = range(10)

for arrival in ARRIVALS:
    print(f"\n===== Mean interarrival: {arrival} ms =====")
    print(
        f"{'Capacity':>8} {'Policy':>10} "
        f"{'Throughput':>11} {'Mean':>9} "
        f"{'P99':>9} {'Long Mean':>11}"
    )

    for capacity in CAPACITIES:
        for policy in ("fcfs", "backfill", "shortest"):
            results = []

            for seed in SEEDS:
                requests = workload(
                    n=500,
                    seed=seed,
                    mean_interarrival=arrival,
                )
                results.append(simulate(
                    requests,
                    policy=policy,
                    kv_capacity=capacity,
                    max_batch=16,
                ))

            def avg(key):
                return statistics.mean(
                    r[key] for r in results
                )

            print(
                f"{capacity:>8} {policy:>10} "
                f"{avg('throughput'):>11.2f} "
                f"{avg('mean_latency'):>9.1f} "
                f"{avg('p99_latency'):>9.1f} "
                f"{avg('long_mean_latency'):>11.1f}"
            )
