import asyncio
import random
import statistics
import time
from dataclasses import dataclass
from collections import deque


NUM_REQUESTS = 200
ARRIVAL_INTERVAL = 0.02

# Service B가 동시에 처리할 수 있는 요청 수
B_CONCURRENCY = 10


@dataclass
class Result:
    request_id: int
    latency: float
    success: bool

class ServiceB:
    def __init__(self):
        self.slots = asyncio.Semaphore(
            B_CONCURRENCY
        )

        self.calls = 0

        self.active = 0
        self.waiting = 0
        self.inflight = 0

        self.max_active = 0
        self.max_waiting = 0
        self.max_inflight = 0

    async def handle(self, request_id):
        self.calls += 1

        self.inflight += 1
        self.waiting += 1

        self.max_inflight = max(
            self.max_inflight,
            self.inflight,
        )

        self.max_waiting = max(
            self.max_waiting,
            self.waiting,
        )

        try:
            async with self.slots:
                self.waiting -= 1
                self.active += 1

                self.max_active = max(
                    self.max_active,
                    self.active,
                )

                try:
                    r = random.random()

                    if r < 0.60:
                        delay = random.uniform(
                            0.02, 0.04
                        )

                    elif r < 0.95:
                        delay = random.uniform(
                            0.8, 1.2
                        )

                    else:
                        delay = random.uniform(
                            0.8, 1.2
                        )

                        await asyncio.sleep(delay)

                        raise RuntimeError(
                            "Service B failed"
                        )

                    await asyncio.sleep(delay)

                    return {
                        "request_id": request_id
                    }

                finally:
                    self.active -= 1

        finally:
            self.inflight -= 1


class CircuitOpenError(Exception):
    pass

class ServiceA:
    def __init__(self, service_b):
        self.service_b = service_b
        self.background_tasks = set()

        # circuit breaker
        self.state = "CLOSED"

        self.results = deque(maxlen=10)

        # 최근 10개 중 50% 이상 실패하면 OPEN
        self.failure_rate_threshold = 0.5
        self.min_requests = 10

        self.opened_at = None
        self.recovery_timeout = 1.0

        self.half_open_inflight = False

        self.rejected = 0

    async def call_b(self, request_id):
        task = asyncio.create_task(
            self.service_b.handle(request_id)
        )

        self.background_tasks.add(task)

        def done_callback(t):
            self.background_tasks.discard(t)

            if not t.cancelled():
                try:
                    t.exception()
                except Exception:
                    pass

        task.add_done_callback(done_callback)

        return await asyncio.wait_for(
            asyncio.shield(task),
            timeout=0.15,
        )

    def allow_request(self):
        now = time.monotonic()

        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if (
                now - self.opened_at
                >= self.recovery_timeout
            ):
                print(
                    "BREAKER: OPEN -> HALF_OPEN"
                )

                self.state = "HALF_OPEN"
                self.half_open_inflight = False

            else:
                self.rejected += 1
                return False

        if self.state == "HALF_OPEN":
            #
            # 시험 요청 하나만 허용
            #
            if self.half_open_inflight:
                self.rejected += 1
                return False

            self.half_open_inflight = True
            return True

        return False

    def open_circuit(self):
        if self.state != "OPEN":
            print("BREAKER: CLOSED -> OPEN")

        self.state = "OPEN"
        self.opened_at = time.monotonic()
        self.half_open_inflight = False


    def record_result(self, success):
        #
        # HALF_OPEN에서는 probe 하나의 결과만 본다.
        #
        if self.state == "HALF_OPEN":
            if success:
                print(
                    "BREAKER: HALF_OPEN -> CLOSED"
                )

                self.state = "CLOSED"
                self.results.clear()
                self.half_open_inflight = False

            else:
                print(
                    "BREAKER: HALF_OPEN -> OPEN"
                )

                self.state = "OPEN"
                self.opened_at = time.monotonic()
                self.half_open_inflight = False

            return

        #
        # CLOSED에서는 최근 결과 window 기록
        #
        self.results.append(success)

        if len(self.results) < self.min_requests:
            return

        failures = sum(
            not result
            for result in self.results
        )

        failure_rate = (
            failures / len(self.results)
        )

        if failure_rate >= self.failure_rate_threshold:
            print(
                f"BREAKER: failure rate "
                f"{failure_rate:.0%}"
            )

            self.open_circuit()

    async def attempt(self, request_id):
        if not self.allow_request():
            raise CircuitOpenError(
                "Circuit breaker is OPEN"
            )

        try:
            result = await self.call_b(
                request_id
            )

            self.record_result(
                success=True
            )

            return result

        except (
            asyncio.TimeoutError,
            RuntimeError,
        ):
            self.record_result(
                success=False
            )

            raise

    async def handle(self, request_id):
        last_error = None

        #
        # 최초 1회 + retry 1회
        #
        for attempt in range(2):
            try:
                return await self.attempt(
                    request_id
                )

            except CircuitOpenError:
                #
                # breaker가 열렸는데
                # retry하는 건 의미 없음.
                #
                raise RuntimeError(
                    "Service unavailable"
                )

            except (
                asyncio.TimeoutError,
                RuntimeError,
            ) as e:
                last_error = e

        raise RuntimeError(
            "Service B failed after retry"
        ) from last_error

class Gateway:
    def __init__(self, service_a):
        self.service_a = service_a

    async def handle(
        self,
        request_id,
        results,
    ):
        start = time.monotonic()

        try:
            await self.service_a.handle(
                request_id
            )

            success = True

        except Exception:
            success = False

        latency = (
            time.monotonic()
            - start
        )

        results.append(
            Result(
                request_id=request_id,
                latency=latency,
                success=success,
            )
        )


async def generate_requests(
    gateway,
    results,
):
    tasks = []

    for request_id in range(
        NUM_REQUESTS
    ):
        task = asyncio.create_task(
            gateway.handle(
                request_id,
                results,
            )
        )

        tasks.append(task)

        await asyncio.sleep(
            ARRIVAL_INTERVAL
        )

    await asyncio.gather(
        *tasks
    )


def percentile(values, p):
    values = sorted(values)

    index = int(
        len(values) * p
    )

    index = min(
        index,
        len(values) - 1,
    )

    return values[index]


async def main():
    random.seed(42)

    service_b = ServiceB()

    service_a = ServiceA(
        service_b
    )

    gateway = Gateway(
        service_a
    )

    results = []

    start = time.monotonic()

    await generate_requests(
        gateway,
        results,
    )

    elapsed = (
        time.monotonic()
        - start
    )

    latencies = [
        r.latency
        for r in results
    ]

    successes = sum(
        r.success
        for r in results
    )

    print()
    print("=== BASELINE ===")

    print(
        f"requests: {len(results)}"
    )

    print(
        f"success: {successes}"
    )

    print(
        f"failed: "
        f"{len(results) - successes}"
    )

    print(
        f"total time: "
        f"{elapsed:.3f}s"
    )

    print(
        f"mean: "
        f"{statistics.mean(latencies):.3f}s"
    )

    print(
        f"median: "
        f"{statistics.median(latencies):.3f}s"
    )

    print(
        f"p95: "
        f"{percentile(latencies, 0.95):.3f}s"
    )

    print(
        f"p99: "
        f"{percentile(latencies, 0.99):.3f}s"
    )

    print(
        f"max: "
        f"{max(latencies):.3f}s"
    )

    print(
        f"downstream calls: "
        f"{service_b.calls}"
    )

    print(
        f"current inflight: "
        f"{service_b.inflight}"
    )

    print(
        f"max inflight: "
        f"{service_b.max_inflight}"
    )

    print(
        f"max waiting: "
        f"{service_b.max_waiting}"
    )

    print(
        f"max active: "
        f"{service_b.max_active}"
    )

    print(
        f"breaker state: "
        f"{service_a.state}"
    )

    print(
        f"breaker rejected: "
        f"{service_a.rejected}"
    )

    print(
    f"breaker state: "
    f"{service_a.state}"
    )

    print(
        f"breaker rejected: "
        f"{service_a.rejected}"
    )


if __name__ == "__main__":
    asyncio.run(main())