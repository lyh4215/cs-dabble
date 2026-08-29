import argparse
import asyncio
import struct
import time
import statistics


HOST = "127.0.0.1"
PORT = 5000


async def send_message(
    reader,
    writer,
    payload,
):
    frame = (
        struct.pack("!I", len(payload))
        + payload
    )

    start = time.perf_counter()

    writer.write(frame)
    await writer.drain()

    header = await reader.readexactly(4)

    (length,) = struct.unpack(
        "!I",
        header,
    )

    response = await reader.readexactly(length)

    end = time.perf_counter()

    if response != payload:
        raise RuntimeError(
            "invalid echo response"
        )

    return end - start


async def worker(
    requests,
    payload,
    latencies,
):
    reader, writer = (
        await asyncio.open_connection(
            HOST,
            PORT,
        )
    )

    try:
        for _ in range(requests):
            latency = await send_message(
                reader,
                writer,
                payload,
            )

            latencies.append(latency)

    finally:
        writer.close()
        await writer.wait_closed()


def percentile(values, p):
    values = sorted(values)

    index = int(
        (len(values) - 1) * p
    )

    return values[index]


async def benchmark(
    concurrency,
    requests_per_connection,
    message_size,
):
    payload = b"x" * message_size
    latencies = []

    start = time.perf_counter()

    tasks = [
        asyncio.create_task(
            worker(
                requests_per_connection,
                payload,
                latencies,
            )
        )
        for _ in range(concurrency)
    ]

    await asyncio.gather(*tasks)

    elapsed = (
        time.perf_counter()
        - start
    )

    total_requests = (
        concurrency
        * requests_per_connection
    )

    throughput = (
        total_requests / elapsed
    )

    latency_ms = [
        x * 1000
        for x in latencies
    ]

    return {
        "connections": concurrency,
        "requests": total_requests,
        "throughput": throughput,
        "mean": statistics.mean(latency_ms),
        "p50": percentile(latency_ms, 0.50),
        "p95": percentile(latency_ms, 0.95),
        "p99": percentile(latency_ms, 0.99),
    }


async def run_sweep(message_size):
    configs = [
        (1, 1000),
        (10, 1000),
        (50, 500),
        (100, 300),
        (500, 100),
    ]

    results = []

    for concurrency, requests in configs:
        print(
            f"running c={concurrency}, "
            f"n={requests}..."
        )

        result = await benchmark(
            concurrency,
            requests,
            message_size,
        )

        results.append(result)

    print()
    print(
        f"{'conn':>6} "
        f"{'req':>8} "
        f"{'msg/s':>10} "
        f"{'mean':>9} "
        f"{'p50':>9} "
        f"{'p95':>9} "
        f"{'p99':>9}"
    )

    print("-" * 75)

    for r in results:
        print(
            f"{r['connections']:>6} "
            f"{r['requests']:>8} "
            f"{r['throughput']:>10.0f} "
            f"{r['mean']:>8.3f}ms "
            f"{r['p50']:>8.3f}ms "
            f"{r['p95']:>8.3f}ms "
            f"{r['p99']:>8.3f}ms"
        )


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=100,
    )

    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=100,
    )

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
    )

    args = parser.parse_args()

    if args.sweep:
        await run_sweep(
            args.size
        )
    else:
        result = await benchmark(
            args.concurrency,
            args.requests,
            args.size,
        )

        print(result)


if __name__ == "__main__":
    asyncio.run(main())