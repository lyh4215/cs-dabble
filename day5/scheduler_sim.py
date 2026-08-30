import asyncio
import random
import statistics
import time
from dataclasses import dataclass
import sys


@dataclass
class Job:
    job_id: int
    created_at: float

class Worker:
    def __init__(self, name, speed, alpha=0.25):
        self.name = name

        self.actual_speed = speed
        self.nominal_speed = speed

        self.queue = asyncio.Queue()
        self.completed = 0
        self.busy = False

        self.estimated_service_time = 1.0 / speed

        self.alpha = alpha

    async def run(self, results):
        while True:
            job = await self.queue.get()

            self.busy = True

            work = random.uniform(
                0.8,
                1.2,
            )

            service_time = (
                work / self.actual_speed
            )

            start = time.monotonic()

            await asyncio.sleep(
                service_time
            )

            end = time.monotonic()

            observed_service_time = (
                end - start
            )

            #
            # Exponential Moving Average
            #

            self.estimated_service_time = (
                self.alpha * observed_service_time
                + (1 - self.alpha)
                * self.estimated_service_time
            )

            latency = (
                end - job.created_at
            )

            results.append({
                "job_id": job.job_id,
                "worker": self.name,
                "latency": latency,
                "service_time": observed_service_time,
                "estimated_service_time":
                    self.estimated_service_time,
            })

            self.completed += 1
            self.busy = False

            self.queue.task_done()


class RoundRobinDispatcher:
    def __init__(self, workers):
        self.workers = workers
        self.index = 0


    def choose_worker(self):
        worker = self.workers[
            self.index
        ]

        self.index = (
            self.index + 1
        ) % len(self.workers)

        return worker

class LoadAwareDispatcher:
    def __init__(self, workers):
        self.workers = workers

    def choose_worker(self):
        return min(
            self.workers,
            key=lambda worker: worker.queue.qsize(),
        )

class SpeedAwareDispatcher:
    def __init__(self, workers):
        self.workers = workers

    def choose_worker(self):
        return min(
            self.workers,
            key=lambda worker:
                (
                    worker.queue.qsize()
                    + int(worker.busy)
                )
                / worker.speed,
        )

class StaticSpeedDispatcher:
    def __init__(self, workers):
        self.workers = workers

    def choose_worker(self):
        return min(
            self.workers,
            key=lambda w:
                (
                    w.queue.qsize()
                    + int(w.busy)
                    + 1
                )
                / w.nominal_speed,
        )

class AdaptiveDispatcher:
    def __init__(self, workers):
        self.workers = workers

    def choose_worker(self):
        return min(
            self.workers,
            key=lambda w:
                (
                    w.queue.qsize()
                    + int(w.busy)
                    + 1
                )
                * w.estimated_service_time,
        )


async def degrade_worker(worker):
    await asyncio.sleep(2.0)

    print()
    print(
        f">>> {worker.name} slowdown: "
        f"{worker.actual_speed} → 1.0"
    )
    print()

    worker.actual_speed = 1.0

async def main():
    mode = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "static"
    )

    alpha = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else 0.25
    )

    workers = [
        Worker(
            "A",
            speed=10.0,
            alpha=alpha
        ),

        Worker(
            "B",
            speed=5.0,
            alpha=alpha
        ),

        Worker(
            "C",
            speed=2.0,
            alpha=alpha
        ),
    ]
    asyncio.create_task(
        degrade_worker(workers[0])
    )

    results = []

    for worker in workers:
        asyncio.create_task(
            worker.run(results)
        )

    if mode == "static":
        dispatcher = StaticSpeedDispatcher(
            workers
        )

    elif mode == "adaptive":
        dispatcher = AdaptiveDispatcher(
            workers
        )

    else:
        raise ValueError(
            "mode must be static or adaptive"
        )
    
    NUM_JOBS = 50
    ARRIVAL_INTERVAL = 0.15


    for job_id in range(
        NUM_JOBS
    ):
        job = Job(
            job_id=job_id,
            created_at=time.monotonic(),
        )

        worker = (
            dispatcher.choose_worker()
        )

        await worker.queue.put(
            job
        )

        await asyncio.sleep(
            ARRIVAL_INTERVAL
        )


    #
    # 모든 worker queue가 비워질 때까지 기다림
    #
    for worker in workers:
        await worker.queue.join()


    latencies = [
        x["latency"]
        for x in results
    ]


    print()
    print("=== RESULT ===")

    print(
        f"jobs: {len(results)}"
    )

    print(
        f"mean latency: "
        f"{statistics.mean(latencies):.3f}s"
    )

    print(
        f"median latency: "
        f"{statistics.median(latencies):.3f}s"
    )

    print(
        f"max latency: "
        f"{max(latencies):.3f}s"
    )

    print()


    for worker in workers:
        worker_results = [
            x
            for x in results
            if x["worker"]
            == worker.name
        ]

        worker_latencies = [
            x["latency"]
            for x in worker_results
        ]

        print(
            f"{worker.name}: "
            f"jobs={len(worker_results)}, "
            f"mean_latency="
            f"{statistics.mean(worker_latencies):.3f}s"
        )


if __name__ == "__main__":
    asyncio.run(
        main()
    )