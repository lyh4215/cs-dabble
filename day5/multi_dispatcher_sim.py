import asyncio
import random
import statistics
import time
from dataclasses import dataclass


@dataclass
class Job:
    job_id: int
    created_at: float
    dispatcher: str


class Worker:
    def __init__(self, name, speed):
        self.name = name
        self.speed = speed
        self.queue = asyncio.Queue()
        self.busy = False

    def load(self):
        return self.queue.qsize() + int(self.busy)

    async def run(self, results):
        while True:
            job = await self.queue.get()

            self.busy = True

            service_time = random.uniform(
                0.8,
                1.2,
            ) / self.speed

            await asyncio.sleep(
                service_time
            )

            latency = (
                time.monotonic()
                - job.created_at
            )

            results.append({
                "job_id": job.job_id,
                "dispatcher": job.dispatcher,
                "worker": self.name,
                "latency": latency,
            })

            self.busy = False
            self.queue.task_done()


class Dispatcher:
    def __init__(
        self,
        name,
        workers,
        refresh_interval,
    ):
        self.name = name
        self.workers = workers
        self.refresh_interval = (
            refresh_interval
        )

        #
        # dispatcher가 들고 있는
        # local scoreboard
        #
        self.view = {
            worker.name: 0
            for worker in workers
        }

    async def refresh_loop(self):
        while True:
            #
            # 실제 worker 상태를
            # 주기적으로 복사
            #
            for worker in self.workers:
                self.view[
                    worker.name
                ] = worker.load()

            await asyncio.sleep(
                self.refresh_interval
            )

    def choose_worker(self):
        min_load = min(
            self.view[w.name]
            for w in self.workers
        )

        candidates = [
            w
            for w in self.workers
            if self.view[w.name] == min_load
        ]

        return random.choice(candidates)

    async def submit(self, job):
        worker = self.choose_worker()

        await worker.queue.put(job)

        #
        # 다음 scoreboard refresh를 기다리지 않고
        # "내가 방금 하나 보냈다"는 사실은
        # 내 local view에 즉시 반영한다.
        #
        self.view[worker.name] += 1


async def generate_jobs(
    dispatcher,
    start_id,
    num_jobs,
    arrival_interval,
):
    for i in range(num_jobs):
        job = Job(
            job_id=start_id + i,
            created_at=time.monotonic(),
            dispatcher=dispatcher.name,
        )

        await dispatcher.submit(
            job
        )

        await asyncio.sleep(
            arrival_interval
        )

async def trace_queues(workers):
    start = time.monotonic()

    while True:
        now = time.monotonic() - start

        print(
            f"{now:5.2f}s "
            f"W1={workers[0].load():2d} "
            f"W2={workers[1].load():2d}"
        )

        await asyncio.sleep(0.1)


async def main():
    workers = [
        Worker(
            "W1",
            speed=5.0,
        ),
        Worker(
            "W2",
            speed=5.0,
        ),
    ]

    asyncio.create_task(
        trace_queues(workers)
    )

    results = []

    for worker in workers:
        asyncio.create_task(
            worker.run(results)
        )

    #
    # 두 dispatcher는
    # 각자 scoreboard를 가진다
    #
    dispatcher_a = Dispatcher(
        "A",
        workers,
        refresh_interval=0.5,
    )

    dispatcher_b = Dispatcher(
        "B",
        workers,
        refresh_interval=0.5,
    )

    asyncio.create_task(
        dispatcher_a.refresh_loop()
    )

    asyncio.create_task(
        dispatcher_b.refresh_loop()
    )

    #
    # 두 dispatcher가 동시에
    # job 생성
    #
    await asyncio.gather(
        generate_jobs(
            dispatcher_a,
            start_id=0,
            num_jobs=50,
            arrival_interval=0.21,
        ),

        generate_jobs(
            dispatcher_b,
            start_id=1000,
            num_jobs=50,
            arrival_interval=0.21,
        ),
    )

    for worker in workers:
        await worker.queue.join()

    latencies = [
        result["latency"]
        for result in results
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
            r
            for r in results
            if r["worker"]
            == worker.name
        ]

        print(
            f"{worker.name}: "
            f"jobs={len(worker_results)}"
        )


if __name__ == "__main__":
    asyncio.run(main())