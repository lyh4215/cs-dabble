
import asyncio
import os
import time

HOST = "127.0.0.1"
PORT = 9000

WORKERS = 4
POLICY = os.getenv("POLICY", "adaptive")
QUEUE_LIMIT = int(os.getenv("QUEUE_LIMIT", "20"))

TARGET_MS = 300
SLOW_AFTER = 1.25

queue = asyncio.Queue()
start_time = None

# Estimated service time, updated after every completion.
service_ewma = 0.05
ALPHA = 0.2

completed = 0
rejected = 0
peak_queue = 0


def service_time():
    elapsed = time.perf_counter() - start_time
    return 0.10 if elapsed >= SLOW_AFTER else 0.05


def predicted_latency_ms(q, estimated_service):
    # Conservative estimate:
    # account for queued requests and own service time.
    batches = (q // WORKERS) + 1
    return batches * estimated_service * 1000


def should_reject():
    q = queue.qsize()

    if POLICY == "fixed":
        return q >= QUEUE_LIMIT

    if POLICY == "adaptive":
        estimated_service = service_ewma

    elif POLICY == "oracle":
        # Experiment-only: access true service time.
        estimated_service = service_time()

    else:
        raise ValueError(f"Unknown policy: {POLICY}")

    predicted = predicted_latency_ms(
        q, estimated_service
    )

    return predicted > TARGET_MS



async def worker():
    global completed, service_ewma

    while True:
        future = await queue.get()
        try:
            start = time.perf_counter()
            await asyncio.sleep(service_time())

            measured = time.perf_counter() - start
            service_ewma = (
                ALPHA * measured
                + (1 - ALPHA) * service_ewma
            )

            completed += 1
            if not future.done():
                future.set_result(None)
        finally:
            queue.task_done()


async def handle_client(reader, writer):
    global rejected, peak_queue

    try:
        data = await reader.readline()
        if not data:
            return

        if should_reject():
            rejected += 1
            writer.write(b"BUSY\n")
            await writer.drain()
            return

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        queue.put_nowait(future)
        peak_queue = max(peak_queue, queue.qsize())

        await future

        writer.write(b"OK\n")
        await writer.drain()

    finally:
        writer.close()
        await writer.wait_closed()


async def monitor():
    while True:
        await asyncio.sleep(0.5)
        print(
            f"t={time.perf_counter()-start_time:.2f}s "
            f"q={queue.qsize()} "
            f"peak={peak_queue} "
            f"service_ewma={service_ewma*1000:.1f}ms "
            f"completed={completed} "
            f"rejected={rejected}",
            flush=True,
        )


async def main():
    global start_time

    for _ in range(WORKERS):
        asyncio.create_task(worker())

    server = await asyncio.start_server(
        handle_client, HOST, PORT
    )
    start_time = time.perf_counter()

    print(
        f"READY policy={POLICY} "
        f"queue_limit={QUEUE_LIMIT}",
        flush=True,
    )

    asyncio.create_task(monitor())

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
