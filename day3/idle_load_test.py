import argparse
import asyncio
import statistics
import struct
import time


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

    response = await reader.readexactly(
        length
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if response != payload:
        raise RuntimeError(
            "invalid echo response"
        )

    return elapsed


async def active_worker(
    reader,
    writer,
    requests,
    payload,
    latencies,
):
    for _ in range(requests):
        latency = await send_message(
            reader,
            writer,
            payload,
        )

        latencies.append(latency)


async def open_connections(
    total,
    batch_size=100,
):
    connections = []

    #
    # 한 번에 수백 개 connect()를 날리면
    # listen backlog나 client scheduler 영향이
    # 커질 수 있으므로 batch 단위로 연결한다.
    #
    for start in range(
        0,
        total,
        batch_size,
    ):
        count = min(
            batch_size,
            total - start,
        )

        batch = await asyncio.gather(
            *[
                asyncio.open_connection(
                    HOST,
                    PORT,
                )
                for _ in range(count)
            ]
        )

        connections.extend(batch)

        #
        # server가 accept할 여유를 조금 준다.
        #
        await asyncio.sleep(0.02)

    return connections


async def close_connections(
    connections,
):
    for _, writer in connections:
        writer.close()

    await asyncio.gather(
        *[
            writer.wait_closed()
            for _, writer in connections
        ],
        return_exceptions=True,
    )


def percentile(
    values,
    p,
):
    values = sorted(values)

    index = int(
        (len(values) - 1) * p
    )

    return values[index]


async def benchmark(
    total_connections,
    active_connections,
    requests_per_active,
    message_size,
):
    if active_connections > total_connections:
        raise ValueError(
            "active_connections "
            "cannot exceed total_connections"
        )

    payload = (
        b"x" * message_size
    )

    #
    # 먼저 모든 TCP connection을 만든다.
    #
    connections = await open_connections(
        total_connections
    )

    try:
        #
        # 여기까지는 benchmark 시간에 포함하지 않는다.
        #
        # 서버 입장에서는 이미:
        #
        #   total_connections 개의 fd가 등록됨
        #
        # 상태다.
        #

        await asyncio.sleep(0.1)

        active = connections[
            :active_connections
        ]

        latencies = []

        start = time.perf_counter()

        tasks = [
            asyncio.create_task(
                active_worker(
                    reader,
                    writer,
                    requests_per_active,
                    payload,
                    latencies,
                )
            )
            for reader, writer in active
        ]

        await asyncio.gather(*tasks)

        elapsed = (
            time.perf_counter()
            - start
        )

        total_requests = (
            active_connections
            * requests_per_active
        )

        throughput = (
            total_requests
            / elapsed
        )

        latency_ms = [
            x * 1000
            for x in latencies
        ]

        return {
            "total": total_connections,
            "active": active_connections,
            "requests": total_requests,

            "throughput": throughput,

            "mean": statistics.mean(
                latency_ms
            ),

            "p50": percentile(
                latency_ms,
                0.50,
            ),

            "p95": percentile(
                latency_ms,
                0.95,
            ),

            "p99": percentile(
                latency_ms,
                0.99,
            ),
        }

    finally:
        await close_connections(
            connections
        )


async def run_sweep(
    active_connections,
    requests_per_active,
    message_size,
    large=False
):
    #
    # select의 FD_SETSIZE를 고려해
    # 공통 비교에서는 800까지만 사용.
    #
    if large:
        totals = [
            1000,
            2000,
            5000,
            10000,
        ]
    else:
        totals = [
            100,
            300,
            500,
            800,
        ]

    results = []

    for total in totals:
        print(
            f"running "
            f"total={total}, "
            f"active={active_connections}..."
        )

        result = await benchmark(
            total,
            active_connections,
            requests_per_active,
            message_size,
        )

        results.append(result)

        await asyncio.sleep(0.2)


    print()

    print(
        f"{'total':>7} "
        f"{'active':>7} "
        f"{'req':>8} "
        f"{'msg/s':>10} "
        f"{'mean':>9} "
        f"{'p50':>9} "
        f"{'p95':>9} "
        f"{'p99':>9}"
    )

    print("-" * 83)

    for r in results:
        print(
            f"{r['total']:>7} "
            f"{r['active']:>7} "
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
        "--total",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--active",
        type=int,
        default=10,
    )

    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=1000,
        help=(
            "requests per active connection"
        ),
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

    parser.add_argument(
        "--large-sweep",
        action="store_true",
    )

    args = parser.parse_args()

    if args.sweep or args.large_sweep:
        await run_sweep(
            args.active,
            args.requests,
            args.size,
            large=args.large_sweep,
        )

    else:
        result = await benchmark(
            args.total,
            args.active,
            args.requests,
            args.size,
        )

        print(result)


if __name__ == "__main__":
    asyncio.run(main())