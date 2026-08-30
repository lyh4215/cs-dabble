import csv

import matplotlib.pyplot as plt


SERVICE_TRACE = "dist/service_trace.csv"
CLIENT_TRACE = "dist/adversarial.csv"


def read_service():
    rows = []

    with open(SERVICE_TRACE) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "timestamp": float(row["timestamp"]),
                "round": int(row["round"]),
                "difficulty": int(row["difficulty"]),
                "queue": int(row["queue"]),
                "peak": int(row["round_peak_queue"]),
            })

    return rows


def read_client():
    rows = []

    with open(CLIENT_TRACE) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "timestamp": float(row["timestamp"]),
                "difficulty": int(row["difficulty"]),
                "solve_ms": float(row["solve_ms"]),
                "total_ms": float(row["total_ms"]),
            })

    return rows


service = read_service()
client = read_client()


#
# 같은 시간축을 만들기 위한 기준점
#
t0 = min(
    service[0]["timestamp"],
    client[0]["timestamp"],
)


service_time = [
    x["timestamp"] - t0
    for x in service
]

client_time = [
    x["timestamp"] - t0
    for x in client
]


#
# 1. Queue
#
plt.figure(figsize=(10, 4))

plt.plot(
    service_time,
    [x["queue"] for x in service],
)

plt.xlabel("Time (s)")
plt.ylabel("Queue size")
plt.title("Service queue over time")
plt.grid()

plt.tight_layout()
plt.savefig("queue.png")
plt.close()


#
# 2. Difficulty
#
plt.figure(figsize=(10, 4))

plt.plot(
    service_time,
    [x["difficulty"] for x in service],
)

plt.xlabel("Time (s)")
plt.ylabel("Puzzle difficulty")
plt.title("Adaptive puzzle difficulty")
plt.grid()

plt.tight_layout()
plt.savefig("difficulty.png")
plt.close()


#
# 3. Normal-client solve latency
#
plt.figure(figsize=(10, 4))

plt.scatter(
    client_time,
    [x["solve_ms"] for x in client],
    s=12,
)

plt.xlabel("Time (s)")
plt.ylabel("Puzzle solve time (ms)")
plt.title("Normal-client puzzle cost")
plt.grid()

plt.tight_layout()
plt.savefig("solve_latency.png")
plt.close()


print("generated:")
print("  queue.png")
print("  difficulty.png")
print("  solve_latency.png")