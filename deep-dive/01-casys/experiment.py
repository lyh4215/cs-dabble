
from scheduler import workload, simulate
import matplotlib.pyplot as plt
import numpy as np

ARRIVALS = [2, 5, 10, 20]
BATCH_SIZES = [1, 2, 4, 8, 16]
POLICIES = ["static", "continuous"]
SEEDS = range(5)

results = []

for arrival in ARRIVALS:
    for batch in BATCH_SIZES:
        for policy in POLICIES:
            metrics = []

            for seed in SEEDS:
                requests = workload(
                    n=500,
                    seed=seed,
                    mean_interarrival=arrival,
                )
                metrics.append(
                    simulate(requests, policy, max_batch=batch)
                )

            results.append({
                "arrival": arrival,
                "batch": batch,
                "policy": policy,
                "throughput": np.mean([
                    m["throughput"] for m in metrics
                ]),
                "mean_latency": np.mean([
                    m["mean_latency"] for m in metrics
                ]),
                "p99_latency": np.mean([
                    m["p99_latency"] for m in metrics
                ]),
            })


fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
metric_names = [
    ("throughput", "Throughput (req/s)"),
    ("mean_latency", "Mean Latency (ms)"),
    ("p99_latency", "P99 Latency (ms)"),
]

for ax, (metric, ylabel) in zip(axes, metric_names):
    for arrival in ARRIVALS:
        for policy in POLICIES:
            subset = [
                r for r in results
                if r["arrival"] == arrival
                and r["policy"] == policy
            ]

            ax.plot(
                [r["batch"] for r in subset],
                [r[metric] for r in subset],
                marker="o" if policy == "continuous" else "x",
                linestyle="-" if policy == "continuous" else "--",
                label=f"{policy}, arrival={arrival}",
            )

    ax.set_xlabel("Max Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title(metric)
    ax.grid(alpha=0.3)

axes[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig("results/sensitivity.png", dpi=150)
plt.show()
