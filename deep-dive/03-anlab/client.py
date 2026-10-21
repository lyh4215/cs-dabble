
import asyncio
import statistics
import sys
import time

HOST = "127.0.0.1"
PORT = 9000


def percentile(values, q):
    values = sorted(values)
    return values[int((len(values) - 1) * q)]


async def request_one():
    start = time.perf_counter()

    try:
        reader, writer = await asyncio.open_connection(
            HOST, PORT
        )

        writer.write(b"REQUEST\n")
        await writer.drain()

        response = (
            await asyncio.wait_for(
                reader.readline(), timeout=30
            )
        ).decode().strip()

        writer.close()
        await writer.wait_closed()

        latency = (time.perf_counter() - start) * 1000

        return response, latency

    except (OSError, asyncio.TimeoutError):
        latency = (time.perf_counter() - start) * 1000
        return "ERROR", latency


async def run(rate, n=300):
    interval = 1 / rate
    tasks = []

    started = time.perf_counter()

    # Open-loop offered load:
    # arrivals do not wait for previous responses.
    for i in range(n):
        target = started + i * interval
        delay = target - time.perf_counter()

        if delay > 0:
            await asyncio.sleep(delay)

        tasks.append(asyncio.create_task(request_one()))

    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - started

    ok = [
        latency
        for response, latency in results
        if response.startswith("OK")
    ]

    busy = sum(
        response == "BUSY"
        for response, _ in results
    )

    errors = sum(
        response == "ERROR"
        for response, _ in results
    )

    print(f"\nOffered rate: {rate} req/s")
    print(f"Total requests: {n}")
    print(f"Completed: {len(ok)}")
    print(f"Rejected: {busy}")
    print(f"Errors: {errors}")
    print(f"Duration: {duration:.2f}s")
    print(f"Goodput: {len(ok)/duration:.2f} req/s")
    print(
        f"Acceptance rate: "
        f"{100 * len(ok) / n:.2f}%"
    )
    print(
        f"Rejection rate: "
        f"{100 * busy / n:.2f}%"
    )

    if ok:
        print(f"Mean latency: {statistics.mean(ok):.2f}ms")
        print(f"P50 latency: {percentile(ok, .50):.2f}ms")
        print(f"P95 latency: {percentile(ok, .95):.2f}ms")
        print(f"P99 latency: {percentile(ok, .99):.2f}ms")



if __name__ == "__main__":
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 40
    asyncio.run(run(rate))
