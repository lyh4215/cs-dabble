import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def cpu_task(
    task_id: int,
    work: int,
    submitted_at: float
):
    started_at = time.perf_counter()

    x = 0.0

    for i in range(1, work):
        x += math.sqrt(i) * math.sin(i)

    finished_at = time.perf_counter()

    return (
        task_id,
        started_at - submitted_at,    # queue delay
        finished_at - started_at,     # execution time
        finished_at - submitted_at,   # total response time
    )


def run_experiment(
    workers: int,
    num_tasks: int,
    work: int,
):
    start = time.perf_counter()

    queue_delays = []
    execution_times = []
    response_times = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []

        for i in range(num_tasks):
            submitted_at = time.perf_counter()

            future = executor.submit(
                cpu_task,
                i,
                work,
                submitted_at,
            )

            futures.append(future)

        for future in as_completed(futures):
            (
                _,
                queue_delay,
                execution_time,
                response_time,
            ) = future.result()

            queue_delays.append(queue_delay)
            execution_times.append(execution_time)
            response_times.append(response_time)

    end = time.perf_counter()

    total_time = end - start
    throughput = num_tasks / total_time

    DEADLINE = 3.0

    misses = sum(
        response_time > DEADLINE
        for response_time in response_times
    )

    miss_rate = misses / num_tasks

    print(f"\nworkers = {workers}")
    print(f"total time         : {total_time:.3f} s")
    print(f"throughput         : {throughput:.2f} tasks/s")
    print(f"mean queue delay   : {statistics.mean(queue_delays):.3f} s")
    print(f"mean execution     : {statistics.mean(execution_times):.3f} s")
    print(f"mean response time : {statistics.mean(response_times):.3f} s")
    print(f"max response time  : {max(response_times):.3f} s")
    print(f"deadline misses    : {misses}/{num_tasks}")
    print(f"deadline miss rate : {miss_rate * 100:.1f}%")

    sorted_response = sorted(response_times)

    print("response times:")
    print(
        "  "
        + " ".join(
            f"{x:.2f}"
            for x in sorted_response
        )
    )

if __name__ == "__main__":
    print(f"os.cpu_count() = {os.cpu_count()}")

    NUM_TASKS = 16
    WORK = 3_000_000

    for workers in [1, 2, 4, 8]:
        run_experiment(
            workers=workers,
            num_tasks=NUM_TASKS,
            work=WORK,
        )